import json
import numpy as np
import os
import webbrowser
from pathlib import Path

from layerbrain import (
    BrainTrainer,
    DentateGranuleConfig,
    NeuromodulationConfig,
    PyramidalConfig,
    SpikingClassifier,
    TextDataset,
    ThalamicConfig,
)


LETTERS = ("H", "O", "L", "A")
END = "<FIN>"
START = "<INICIO>"
CLASSES = LETTERS + (END,)
CONTEXTS = LETTERS + (START,)


def build_visualization_payload(model, frames):
    layers = []
    incoming_counts = [
        np.zeros(layer.num_neurons, dtype=np.int32)
        for layer in model.network.layers
    ]
    for _, post_idx, synapse in model.network.connections:
        if hasattr(synapse, "post_indices"):
            incoming_counts[post_idx] += np.bincount(
                synapse.post_indices,
                minlength=model.network.layers[post_idx].num_neurons,
            )
        elif hasattr(synapse, "W"):
            incoming_counts[post_idx] += np.count_nonzero(synapse.W, axis=1)
    for index, layer in enumerate(model.network.layers):
        if index == 0 and model.sensor_dataset is not None:
            name = f"Entrada · {model.sensor_dataset.modality}"
        elif index == model.output_layer:
            name = "Salida · predicción"
        else:
            name = type(layer).__name__.removesuffix("Layer")
        layers.append(
            {
                "name": name,
                "size": layer.num_neurons,
                "num_dendrites": layer.num_dendrites,
                "max_incoming": int(np.max(incoming_counts[index])),
            }
        )

    connections = []
    for connection_index, (_, _, synapse) in enumerate(model.network.connections):
        if hasattr(synapse, "pre_indices"):
            pre_indices = synapse.pre_indices
            post_indices = synapse.post_indices
        elif hasattr(synapse, "W"):
            post_indices, pre_indices = np.nonzero(synapse.W)
        else:
            continue
        for pre_neuron, post_neuron in zip(pre_indices, post_indices):
            connections.append(
                [
                    model.network.connections[connection_index][0],
                    int(pre_neuron),
                    model.network.connections[connection_index][1],
                    int(post_neuron),
                    connection_index,
                ]
            )

    return {
        "layers": layers,
        "neuron_count": sum(layer["size"] for layer in layers),
        "synapse_count": len(connections),
        "connections": connections,
        "frames": frames,
    }


def write_visualization(model, frames):
    project_dir = Path(__file__).resolve().parent
    template_path = project_dir / "visualization_template.html"
    output_path = project_dir / "network_activity.html"
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"No se pudo leer la plantilla de visualización {template_path}: {exc}"
        ) from exc

    payload = json.dumps(
        build_visualization_payload(model, frames),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    if "__NETWORK_DATA__" not in template:
        raise RuntimeError(
            f"La plantilla {template_path} no contiene el marcador de datos requerido."
        )
    try:
        output_path.write_text(
            template.replace("__NETWORK_DATA__", payload),
            encoding="utf-8",
        )
    except OSError as exc:
        raise RuntimeError(
            f"No se pudo guardar la visualización HTML en {output_path}: {exc}"
        ) from exc
    return output_path


def main():
    np.random.seed(1)

    # 1. Dataset de texto: contexto anterior y siguiente token esperado.
    dataset = TextDataset(
        samples=[
            (START, "H"),
            ("H", "O"),
            ("O", "L"),
            ("L", "A"),
            ("A", END),
        ],
        prediction_samples=(START, "H", "O", "L", "A"),
        classes=CLASSES,
        encoding_parameters={"max_current": 80.0},
        sensor_parameters={
            "v_thresh": -52.0,
            "t_ref": 2.0,
            "noise_std": 0.0,
            "lateral_inhibition": 0.0,
        },
    )

    # 2. La capa sensorial de texto se conecta automáticamente desde el dataset.
    model = SpikingClassifier(
        dataset=dataset,
        hidden_layers=(
            ThalamicConfig(
                8,
                tau_m=10.0,
                noise_std=0.05,
                gate_threshold=0.08,
                lateral_inhibition=3.0,
                num_dendrites=5,
            ),
            DentateGranuleConfig(
                12,
                max_active_ratio=0.25,
                noise_std=0.05,
                lateral_inhibition=12.0,
                num_dendrites=4,
            ),
            PyramidalConfig(
                12,
                sfa_strength=4.0,
                tau_w=90.0,
                noise_std=0.05,
                lateral_inhibition=4.0,
                num_dendrites=5,
            ),
        ),
        neuromodulation=NeuromodulationConfig(
            tau_ACh=80.0,
            tau_DA=150.0,
            tau_GABA=250.0,
            tau_OXT=400.0,
            gaba_baseline=0.12,
        ),
        output_parameters={"v_thresh": -52.0, "num_dendrites": 10},
        steps_per_sample=30,
        learning_rate=0.35,
        adaptive_routing=True,
        route_exploration=0.05,
        synapse_density=0.25,
    )

    # Conserva una muestra espaciada de los disparos, no el estado completo del modelo.
    total_steps = 60 * len(dataset) * model.steps_per_sample
    frame_stride = max(1, int(np.ceil(total_steps / 1200)))
    activity_frames = [{"creation": True, "spikes": {}, "active_connections": []}]

    def record_activity(frame):
        if frame["frame"] % frame_stride == 0 or frame["frame"] == total_steps - 1:
            activity_frames.append(frame)

    # 3. BrainTrainer entrena con las entradas codificadas por el dataset.
    trainer = BrainTrainer(model.network)
    history = trainer.fit_dataset(
        dataset,
        epochs=60,
        steps_per_sample=model.steps_per_sample,
        learning_rate=model.learning_rate,
        activity_callback=record_activity,
        lineal=False,
    )

    visualization_path = write_visualization(model, activity_frames)
    print(f"Visualización 3D guardada en: {visualization_path}")
    if os.environ.get("LAYERBRAIN_OPEN_VISUALIZATION", "1") != "0":
        opened = webbrowser.open(visualization_path.as_uri())
        if not opened:
            print("No se pudo abrir un navegador automáticamente; abre el HTML indicado.")

    # Comprime los pesos ya entrenados antes de evaluar la inferencia.
    float_weight_bytes = sum(
        synapse.weights.nbytes
        for _, _, synapse in model.network.connections
        if hasattr(synapse, "weights")
    )
    model.quantize_for_inference()
    quantized_weight_bytes = sum(
        synapse.weights.nbytes
        for _, _, synapse in model.network.connections
        if hasattr(synapse, "weights")
    )
    print(
        "Pesos comprimidos para inferencia: "
        f"{quantized_weight_bytes} bytes "
        f"(antes: {float_weight_bytes} bytes; precisión de 4 decimales)"
    )

    # Dataset también guarda entradas sin etiqueta para inferencia por lotes.
    batch_predictions = model.predict_dataset()
    generated = []
    spike_history = []
    for prediction in batch_predictions:
        spike_history.append(prediction.spike_counts)
        if prediction.label == END:
            break
        generated.append(prediction.label)

    word = "".join(generated)
    print(f"Palabra generada: {word or '(vacía)'}")
    print(f"Exactitud en las transiciones entrenadas: {history['accuracy'][-1]:.0%}")
    print(f"Pasos medios por muestra en la última época: {history['mean_steps'][-1]:.1f}/{model.steps_per_sample}")
    print(f"Atención media de ruta en la última época: {history['mean_route_attention'][-1]:.2f}")
    if history.get("best_route") is not None:
        print(
            "Ruta más utilizada: "
            f"{history['best_route']} "
            f"({history['best_route_uses']} usos, "
            f"{history['best_route_mean_steps']:.1f} pasos medios)"
        )
    if history["stopped_early"]:
        print(
            "Entrenamiento detenido al estabilizarse la ruta dominante "
            f"en la época {len(history['accuracy'])}."
        )
    if history.get("best_response_steps") is not None:
        print(
            f"Respuesta más rápida observada: "
            f"{history['best_response_steps']:.1f} pasos"
        )
    print(f"Pérdida: {history['loss'][0]:.3f} -> {history['loss'][-1]:.3f}")
    print(f"Memoria de arrays de la red: {model.network.memory_usage_bytes / 1024:.1f} KiB")
    for previous, predicted, spikes in zip(
        CONTEXTS[-1:] + LETTERS,
        generated + [END],
        spike_history,
    ):
        print(f"{previous} -> {predicted}; impulsos={list(spikes)}")

if __name__ == "__main__":
    main()
