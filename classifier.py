"""API de alto nivel para entrenar y consultar un clasificador SNN."""

from dataclasses import dataclass

import numpy as np

from .data import SensoryDataset, SpikeDataset
from .errors import (
    ArchitectureError,
    DatasetError,
    IncompatibleDimensionsError,
    PersistenceError,
    PredictionError,
    TrainingError,
)
from .network import BrainNetwork
from .neuromodulation import NeuromodulatorySystem
from .neurons import (
    DentateGranuleLayer,
    LIFLayer,
    PyramidalLayer,
    StriatalMSNLayer,
    ThalamicLayer,
)
from .synapses import SparseReservoirSynapse
from .training import BrainTrainer


class NeuronLayerConfig:
    """Configuración reutilizable para una capa neuronal especializada."""

    layer_type = LIFLayer

    def __init__(self, num_neurons, **parameters):
        if not isinstance(num_neurons, (int, np.integer)) or num_neurons <= 0:
            raise ArchitectureError("`num_neurons` debe ser un entero mayor que cero.")
        self.num_neurons = int(num_neurons)
        self.parameters = dict(parameters)

    def build(self):
        try:
            layer = self.layer_type(self.num_neurons, **self.parameters)
        except (TypeError, ValueError) as exc:
            raise ArchitectureError(
                f"Configuración inválida para {self.layer_type.__name__}: {exc}"
            ) from exc
        if not isinstance(layer, LIFLayer):
            raise ArchitectureError(
                f"{self.layer_type.__name__} debe producir una capa compatible con LIFLayer."
            )
        return layer


class LIFConfig(NeuronLayerConfig):
    """Capa LIF configurable mediante ``tau_m``, umbral, ruido, etc."""

    layer_type = LIFLayer


class ThalamicConfig(NeuronLayerConfig):
    """Capa talámica con compuerta de atención, ráfagas y adaptación sensorial."""

    layer_type = ThalamicLayer


class DentateGranuleConfig(NeuronLayerConfig):
    """Capa dispersa k-WTA para separación de patrones."""

    layer_type = DentateGranuleLayer


class PyramidalConfig(NeuronLayerConfig):
    """Capa piramidal con adaptación de frecuencia de disparo."""

    layer_type = PyramidalLayer


class StriatalMSNConfig(NeuronLayerConfig):
    """Capa estriatal con modulación dopaminérgica D1 y selección de acción."""

    layer_type = StriatalMSNLayer


@dataclass(frozen=True)
class NeuromodulationConfig:
    """Parámetros de las señales neuromoduladoras compartidas por la red."""

    dt: float = 1.0
    tau_ACh: float = 100.0
    tau_DA: float = 200.0
    tau_GABA: float = 300.0
    tau_OXT: float = 400.0
    gaba_baseline: float = 0.2
    route_attention_gain: float = 0.8

    def build(self):
        try:
            return NeuromodulatorySystem(
                dt=self.dt,
                tau_ACh=self.tau_ACh,
                tau_DA=self.tau_DA,
                tau_GABA=self.tau_GABA,
                tau_OXT=self.tau_OXT,
                gaba_baseline=self.gaba_baseline,
                route_attention_gain=self.route_attention_gain,
            )
        except (TypeError, ValueError) as exc:
            raise ArchitectureError(f"Configuración neuromoduladora inválida: {exc}") from exc


@dataclass(frozen=True)
class Prediction:
    """Etiqueta ganadora y actividad observada en las neuronas de salida."""

    label: str
    index: int
    spike_counts: tuple[int, ...]


class SpikingClassifier:
    """Clasificador SNN sencillo: declara tamaños, entrena, y usa ``predict``.

    ``hidden_sizes`` define el tamaño de cada capa oculta. Se añade una conexión
    directa de entrada a salida para que el readout pueda aprender aun cuando
    las capas ocultas aleatorias todavía no representen bien las entradas.
    """

    def __init__(
        self,
        input_size=None,
        classes=None,
        hidden_sizes=(),
        steps_per_sample=30,
        learning_rate=0.35,
        hidden_layers=None,
        neuromodulation=None,
        input_parameters=None,
        output_parameters=None,
        dataset=None,
        adaptive_routing=True,
        route_exploration=0.15,
        synapse_density=0.25,
    ):
        if dataset is not None:
            if not isinstance(dataset, SpikeDataset):
                raise ArchitectureError("`dataset` debe ser una instancia de SpikeDataset.")
            if input_size is not None and input_size != dataset.input_size:
                raise ArchitectureError(
                    "`input_size` no coincide con el tamaño de entrada del dataset."
                )
            if classes is not None:
                try:
                    supplied_classes = tuple(classes)
                except TypeError as exc:
                    raise ArchitectureError(
                        "`classes` debe ser una secuencia de etiquetas."
                    ) from exc
                if supplied_classes != dataset.classes:
                    raise ArchitectureError(
                        "`classes` debe coincidir en orden y contenido con el dataset."
                    )
            input_size = dataset.input_size
            classes = dataset.classes
        if input_size is None or classes is None:
            raise ArchitectureError(
                "Proporciona `input_size` y `classes`, o pasa `dataset` para inferirlos."
            )
        if not isinstance(input_size, (int, np.integer)) or input_size <= 0:
            raise ArchitectureError("`input_size` debe ser un entero mayor que cero.")
        try:
            self.classes = tuple(classes)
        except TypeError as exc:
            raise ArchitectureError("`classes` debe ser una secuencia de etiquetas.") from exc
        if len(self.classes) < 2:
            raise ArchitectureError("`classes` debe contener al menos dos etiquetas.")
        if any(not isinstance(label, str) or not label for label in self.classes):
            raise ArchitectureError("Cada clase debe ser texto no vacío.")
        if len(set(self.classes)) != len(self.classes):
            raise ArchitectureError("Las etiquetas de `classes` deben ser únicas.")
        if hidden_layers is not None and hidden_sizes:
            raise ArchitectureError(
                "Usa `hidden_layers` o `hidden_sizes`, no ambos."
            )
        if hidden_layers is not None:
            try:
                layer_configs = tuple(hidden_layers)
            except TypeError as exc:
                raise ArchitectureError("`hidden_layers` debe ser una secuencia de configuraciones.") from exc
            if any(not isinstance(config, NeuronLayerConfig) for config in layer_configs):
                raise ArchitectureError(
                    "`hidden_layers` debe contener configuraciones como "
                    "ThalamicConfig(...) o PyramidalConfig(...)."
                )
        else:
            try:
                sizes = tuple(hidden_sizes)
            except TypeError as exc:
                raise ArchitectureError("`hidden_sizes` debe ser una secuencia de enteros.") from exc
            if any(not isinstance(size, (int, np.integer)) or size <= 0 for size in sizes):
                raise ArchitectureError("Cada tamaño oculto debe ser un entero mayor que cero.")
            layer_configs = tuple(
                LIFConfig(size, noise_std=0.05, lateral_inhibition=0.0)
                for size in sizes
            )
        if neuromodulation is not None and not isinstance(
            neuromodulation, (NeuromodulationConfig, NeuromodulatorySystem)
        ):
            raise ArchitectureError(
                "`neuromodulation` debe ser NeuromodulationConfig o NeuromodulatorySystem."
            )
        for name, parameters in (
            ("input_parameters", input_parameters),
            ("output_parameters", output_parameters),
        ):
            if parameters is not None and not isinstance(parameters, dict):
                raise ArchitectureError(f"`{name}` debe ser un diccionario de parámetros LIF.")
        if (
            not isinstance(steps_per_sample, (int, np.integer))
            or steps_per_sample <= 0
        ):
            raise ArchitectureError("`steps_per_sample` debe ser un entero mayor que cero.")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ArchitectureError("`learning_rate` debe ser finito y mayor que cero.")
        if not isinstance(adaptive_routing, bool):
            raise ArchitectureError("`adaptive_routing` debe ser booleano.")
        if not np.isfinite(route_exploration) or not 0.0 <= route_exploration <= 1.0:
            raise ArchitectureError("`route_exploration` debe estar entre cero y uno.")
        if not np.isfinite(synapse_density) or not 0.0 < synapse_density <= 1.0:
            raise ArchitectureError("`synapse_density` debe estar en (0, 1].")

        self.input_size = int(input_size)
        self.steps_per_sample = int(steps_per_sample)
        self.learning_rate = float(learning_rate)
        self.adaptive_routing = adaptive_routing
        self.route_exploration = float(route_exploration)
        self.synapse_density = float(synapse_density)
        self.sensor_dataset = dataset if isinstance(dataset, SensoryDataset) else None
        self.sensor_signature = (
            self.sensor_dataset.sensor_signature
            if self.sensor_dataset is not None
            else None
        )
        if isinstance(neuromodulation, NeuromodulationConfig):
            neuromodulation = neuromodulation.build()
        self.network = BrainNetwork(neuromodulation=neuromodulation)

        if self.sensor_dataset is not None:
            try:
                input_neurons = self.sensor_dataset.build_sensor_layer(input_parameters)
            except DatasetError as exc:
                raise ArchitectureError(str(exc)) from exc
            if input_neurons.num_neurons != self.input_size:
                raise ArchitectureError(
                    "La capa sensorial debe tener el mismo tamaño que la entrada codificada."
                )
        else:
            input_config = {"noise_std": 0.0, "lateral_inhibition": 0.0}
            input_config.update(input_parameters or {})
            try:
                input_neurons = LIFLayer(self.input_size, **input_config)
            except (TypeError, ValueError) as exc:
                raise ArchitectureError(f"Parámetros de entrada LIF inválidos: {exc}") from exc
        input_layer = self.network.add_layer(input_neurons)
        previous_layer = input_layer
        previous_size = self.input_size
        self.synapses = []

        self.hidden_layers = []
        for config in layer_configs:
            current_neurons = config.build()
            current_layer = self.network.add_layer(current_neurons)
            synapse = SparseReservoirSynapse(
                previous_size,
                current_neurons.num_neurons,
                density=self.synapse_density,
                weight_scale=35.0,
                max_row_sum=2.0,
                num_dendrites=current_neurons.num_dendrites,
            )
            self.network.connect(previous_layer, current_layer, synapse)
            self.synapses.append(synapse)
            self.hidden_layers.append(current_neurons)
            previous_layer = current_layer
            previous_size = current_neurons.num_neurons

        output_config = {"noise_std": 0.0, "lateral_inhibition": 0.0}
        output_config.update(output_parameters or {})
        try:
            output_neurons = LIFLayer(len(self.classes), **output_config)
        except (TypeError, ValueError) as exc:
            raise ArchitectureError(f"Parámetros de salida LIF inválidos: {exc}") from exc
        self.output_layer = self.network.add_layer(output_neurons)
        readout_synapses = []
        if layer_configs:
            readout_synapses.append(
                (
                    previous_layer,
                    SparseReservoirSynapse(
                        previous_size,
                        len(self.classes),
                        density=max(self.synapse_density, 0.5),
                        weight_scale=35.0,
                        max_row_sum=2.0,
                        num_dendrites=output_neurons.num_dendrites,
                    ),
                )
            )
        readout_synapses.append(
            (
                input_layer,
                SparseReservoirSynapse(
                    self.input_size,
                    len(self.classes),
                    density=max(self.synapse_density, 0.5),
                    weight_scale=100.0,
                    max_row_sum=2.0,
                    num_dendrites=output_neurons.num_dendrites,
                ),
            )
        )
        for source_layer, synapse in readout_synapses:
            self.network.connect(source_layer, self.output_layer, synapse)
            self.synapses.append(synapse)

        self.trainer = BrainTrainer(self.network)
        self.history = None

    def _validate_dataset(self, dataset, parameter_name):
        if not isinstance(dataset, SpikeDataset):
            raise TrainingError(
                f"`{parameter_name}` debe ser SpikeDataset; consulta `layerbrain.errors` "
                "si necesitas corregir el formato."
            )
        if dataset.input_size != self.input_size:
            raise IncompatibleDimensionsError(
                f"El dataset tiene entradas de tamaño {dataset.input_size}, pero "
                f"la arquitectura espera {self.input_size}."
            )
        if dataset.classes != self.classes:
            raise TrainingError(
                "El orden y los nombres de las clases del dataset deben coincidir "
                "exactamente con los de la arquitectura."
            )
        if dataset.sensor_signature != self.sensor_signature:
            raise TrainingError(
                "El tipo de dataset sensorial y su configuración deben coincidir "
                "con los usados al crear el clasificador."
            )

    def fit(
        self,
        dataset,
        epochs=20,
        validation_data=None,
        adaptive_routing=None,
        route_exploration=None,
        route_callback=None,
    ):
        """Entrena el readout y devuelve pérdida/exactitud por época."""
        self._validate_dataset(dataset, "dataset")
        if validation_data is not None:
            self._validate_dataset(validation_data, "validation_data")
        if not isinstance(epochs, (int, np.integer)) or epochs <= 0:
            raise TrainingError("`epochs` debe ser un entero mayor que cero.")

        samples = [
            {
                "inputs": {0: features},
                "target_layer": self.output_layer,
                "target_neuron": int(target),
            }
            for features, target in zip(dataset.features, dataset.targets)
        ]
        validation_samples = None
        if validation_data is not None:
            validation_samples = [
                {
                    "inputs": {0: features},
                    "target_layer": self.output_layer,
                    "target_neuron": int(target),
                }
                for features, target in zip(
                    validation_data.features, validation_data.targets
                )
            ]

        self.history = self.trainer.fit_supervised(
            samples,
            epochs=int(epochs),
            steps_per_sample=self.steps_per_sample,
            learning_rate=self.learning_rate,
            validation_data=validation_samples,
            adaptive_routing=(
                self.adaptive_routing
                if adaptive_routing is None
                else adaptive_routing
            ),
            route_exploration=(
                self.route_exploration
                if route_exploration is None
                else route_exploration
            ),
            route_callback=route_callback,
        )
        return self.history

    def predict(self, features):
        """Predice una clase y devuelve también los conteos de impulsos.

        Con un dataset sensorial, ``features`` es una entrada cruda de esa
        modalidad; con ``SpikeDataset`` se espera el vector numérico habitual.
        """
        if self.sensor_dataset is not None:
            try:
                values = self.sensor_dataset.encode_input(features)
            except IncompatibleDimensionsError:
                raise
            except DatasetError as exc:
                raise PredictionError(str(exc)) from exc
        else:
            try:
                values = np.asarray(features, dtype=float)
            except (TypeError, ValueError) as exc:
                raise PredictionError(
                    "`predict` requiere un vector numérico unidimensional."
                ) from exc
            if values.shape != (self.input_size,):
                raise IncompatibleDimensionsError(
                    f"`predict` recibió forma {values.shape}; se esperaba "
                    f"({self.input_size},)."
                )
        if not np.all(np.isfinite(values)):
            raise PredictionError("La entrada de `predict` contiene NaN o infinito.")

        route = (
            self.network.select_route(
                self.output_layer,
                source_layers={0},
            )
            if self.adaptive_routing
            else None
        )
        index, spike_counts = self.network.predict(
            {0: values},
            output_layer_idx=self.output_layer,
            steps=self.steps_per_sample,
            active_connections=route,
        )
        if index < 0:
            if self.network.lineal_mode is True:
                raise PredictionError(
                    "Ninguna neurona de salida disparó en el modo lineal. "
                    "La ruta ya no tiene conexiones de salto; verifica que las "
                    "capas ocultas reciban suficiente corriente para disparar, "
                    "ajusta sus umbrales o aumenta `steps_per_sample`."
                )
            raise PredictionError(
                "Ninguna neurona de salida disparó; revisa escalas, pesos "
                "o `steps_per_sample`."
            )
        return Prediction(
            label=self.classes[index],
            index=index,
            spike_counts=tuple(int(count) for count in spike_counts),
        )

    def quantize_for_inference(self):
        """Reduce sparse weight storage to uint8 for memory-efficient inference."""
        self.network.quantize_for_inference()

    def dequantize_weights(self):
        """Restore float32 sparse weights before manual editing or fine-tuning."""
        self.network.dequantize_weights()

    def _persistence_metadata(self):
        return {
            "input_size": self.input_size,
            "classes": list(self.classes),
            "output_layer": self.output_layer,
            "steps_per_sample": self.steps_per_sample,
            "learning_rate": self.learning_rate,
            "adaptive_routing": self.adaptive_routing,
            "route_exploration": self.route_exploration,
            "synapse_density": self.synapse_density,
            "sensor_signature": repr(self.sensor_signature),
        }

    def save(self, filepath="layerbrain_model.npz"):
        """Guarda los pesos y la topología entrenada del clasificador."""
        return self.network.save_brain(
            filepath,
            model_metadata=self._persistence_metadata(),
        )

    def load(self, filepath="layerbrain_model.npz"):
        """Restaura un modelo guardado en una arquitectura equivalente."""
        metadata = self.network.read_brain_metadata(filepath)
        expected_model = self._persistence_metadata()
        if metadata is not None:
            if metadata.get("model") != expected_model:
                raise PersistenceError(
                    "La configuración del clasificador no coincide con el modelo "
                    "guardado (dimensiones, clases o parámetros de predicción)."
                )
            expected_layers = self.network._persistence_metadata(
                expected_model
            )["layers"]
            if metadata["layers"] != expected_layers:
                raise PersistenceError(
                    "Las capas del clasificador no coinciden con el modelo guardado."
                )
            lineal_mode = metadata.get("lineal_mode")
            if isinstance(lineal_mode, bool) and lineal_mode != self.network.lineal_mode:
                self.trainer._configure_connectivity(lineal_mode)
        self.network.load_brain(
            filepath,
            model_metadata=expected_model,
        )
        return self

    def predict_dataset(self, dataset=None):
        """Predice todas las entradas sin etiqueta guardadas en un dataset."""
        if dataset is None:
            dataset = self.sensor_dataset
        if dataset is None:
            raise PredictionError(
                "Pasa un dataset con `prediction_samples` para predecir un lote."
            )
        self._validate_dataset(dataset, "dataset")
        if not dataset.prediction_inputs:
            raise PredictionError(
                "El dataset no tiene entradas pendientes; configúralas con "
                "`prediction_samples`."
            )
        return tuple(self.predict(item) for item in dataset.prediction_inputs)
