import numpy as np

from .errors import ArchitectureError, DatasetError, TrainingError
from .network import BrainNetwork
from .synapses import SparseReservoirSynapse


class RouteConvergenceCallback:
    """Detiene el entrenamiento cuando la ruta y su velocidad se estabilizan."""

    def __init__(self, patience=10, min_improvement=0.5):
        if (
            not isinstance(patience, (int, np.integer))
            or isinstance(patience, bool)
            or patience <= 0
        ):
            raise TrainingError("`patience` debe ser un entero mayor que cero.")
        if not np.isfinite(min_improvement) or min_improvement < 0.0:
            raise TrainingError("`min_improvement` debe ser finito y no negativo.")
        self.patience = int(patience)
        self.min_improvement = float(min_improvement)
        self._previous_route = None
        self._stable_epochs = 0
        self._best_response_steps = float("inf")
        self._speed_reference_steps = float("inf")
        self._speed_stable_epochs = 0
        self._route_uses = {}
        self._route_steps = {}
        self.best_route = None
        self.best_route_uses = 0
        self.best_route_mean_steps = 0.0
        self.best_response_steps = float("inf")

    def __call__(self, summary):
        """Registra rutas por época y devuelve si se debe detener el entrenamiento."""
        routes = summary["routes"]
        if not routes:
            return False

        for route_info in routes:
            route = tuple(route_info["route"])
            uses = route_info["uses"]
            self._route_uses[route] = self._route_uses.get(route, 0) + uses
            self._route_steps[route] = (
                self._route_steps.get(route, 0.0)
                + uses * route_info["mean_steps"]
            )

        dominant = min(
            routes,
            key=lambda item: (
                -item["uses"],
                item["mean_steps"],
                tuple(item["route"]),
            ),
        )
        dominant_route = tuple(dominant["route"])
        if dominant_route == self._previous_route:
            self._stable_epochs += 1
        else:
            self._previous_route = dominant_route
            self._stable_epochs = 1

        response_steps = summary["mean_steps"]
        self._best_response_steps = min(self._best_response_steps, response_steps)
        if response_steps < self._speed_reference_steps - self.min_improvement:
            self._speed_reference_steps = response_steps
            self._speed_stable_epochs = 0
        else:
            self._speed_stable_epochs += 1
        self.best_response_steps = self._best_response_steps

        ranked_routes = sorted(
            self._route_uses,
            key=lambda route: (
                -self._route_uses[route],
                self._route_steps[route] / self._route_uses[route],
                route,
            ),
        )
        self.best_route = ranked_routes[0]
        self.best_route_uses = self._route_uses[self.best_route]
        self.best_route_mean_steps = (
            self._route_steps[self.best_route] / self.best_route_uses
        )
        summary["dominant_route"] = list(dominant_route)
        summary["best_route"] = list(self.best_route)
        summary["best_route_uses"] = self.best_route_uses
        summary["best_route_mean_steps"] = self.best_route_mean_steps
        summary["best_response_steps"] = self._best_response_steps
        summary["stable_route_epochs"] = self._stable_epochs
        summary["stable_speed_epochs"] = self._speed_stable_epochs
        return (
            self._stable_epochs >= self.patience
            and self._speed_stable_epochs >= self.patience
        )


class BrainTrainer:
    def __init__(self, network: BrainNetwork):
        self.network = network

    def _apply_linear_supervision(
        self,
        route,
        output_error,
        output_layer,
        layer_activity,
        sample_steps,
        learning_rate,
    ):
        error_by_layer = {output_layer: output_error}
        connection_indices = (
            range(len(self.network.connections))
            if route is None
            else route
        )
        ordered_connections = sorted(
            connection_indices,
            key=lambda index: self.network.connections[index][1],
            reverse=True,
        )
        for index in ordered_connections:
            pre_idx, post_idx, synapse = self.network.connections[index]
            post_error = error_by_layer.get(post_idx)
            if post_error is None:
                continue
            pre_rate = layer_activity[pre_idx] / sample_steps
            if hasattr(synapse, "apply_supervised_update"):
                synapse.apply_supervised_update(
                    post_error,
                    pre_rate,
                    learning_rate,
                )
                propagated_error = np.bincount(
                    synapse.pre_indices,
                    weights=(
                        synapse.weights
                        * post_error[synapse.post_indices]
                    ),
                    minlength=synapse.pre_size,
                )
            elif hasattr(synapse, "W"):
                mask = getattr(synapse, "connectivity_mask", 1.0)
                synapse.W += (
                    learning_rate * np.outer(post_error, pre_rate) * mask
                )
                synapse.W[:] = np.clip(synapse.W, 0.0, 1.2) * mask
                propagated_error = synapse.W.T @ post_error
            else:
                continue
            magnitude = float(np.max(np.abs(propagated_error)))
            if magnitude > 1.0:
                propagated_error /= magnitude
            error_by_layer[pre_idx] = propagated_error

    def _configure_connectivity(self, lineal):
        if not isinstance(lineal, bool):
            raise TrainingError("`lineal` debe ser booleano.")
        self.network.lineal_mode = lineal

        allowed_pairs = (
            {
                (index, index + 1)
                for index in range(len(self.network.layers) - 1)
            }
            if lineal
            else {
                (source, target)
                for source in range(len(self.network.layers))
                for target in range(len(self.network.layers))
            }
        )
        connections_by_pair = {}
        for source, target, synapse in self.network.connections:
            if (source, target) in allowed_pairs:
                connections_by_pair.setdefault((source, target), []).append(
                    synapse
                )

        connections = []
        route_scores = []
        for source, target in sorted(allowed_pairs):
            synapses = connections_by_pair.get((source, target))
            if synapses:
                connections.extend(
                    (source, target, synapse) for synapse in synapses
                )
                route_scores.extend(
                    self.network.route_scores[index]
                    for index, (old_source, old_target, _) in enumerate(
                        self.network.connections
                    )
                    if old_source == source and old_target == target
                )
            else:
                pre_layer = self.network.layers[source]
                post_layer = self.network.layers[target]
                synapse = SparseReservoirSynapse(
                    pre_layer.num_neurons,
                    post_layer.num_neurons,
                    density=0.25,
                    weight_scale=35.0,
                    max_row_sum=2.0,
                    num_dendrites=post_layer.num_dendrites,
                )
                connections.append((source, target, synapse))
                route_scores.append(0.0)

        self.network.connections = connections
        self.network.route_scores = route_scores
        self.network._incoming_connections = {}
        for index, (_, target, _) in enumerate(connections):
            self.network._incoming_connections.setdefault(target, []).append(
                index
            )
        self.network.enforce_dendrite_limits()
        self.network.reset_traces()

    def fit_supervised(
        self,
        dataset,
        epochs=10,
        steps_per_sample=30,
        learning_rate=0.2,
        validation_data=None,
        adaptive_routing=True,
        route_exploration=0.15,
        activity_callback=None,
        route_callback=None,
        lineal=True,
    ):
        if not dataset:
            raise DatasetError("El dataset de entrenamiento está vacío.")
        if epochs <= 0 or steps_per_sample <= 0:
            raise TrainingError("epochs y steps_per_sample deben ser mayores que cero.")
        if learning_rate <= 0:
            raise TrainingError("learning_rate debe ser mayor que cero.")
        if not isinstance(adaptive_routing, bool):
            raise TrainingError("`adaptive_routing` debe ser booleano.")
        if not isinstance(lineal, bool):
            raise TrainingError("`lineal` debe ser booleano.")
        if not np.isfinite(route_exploration) or not 0.0 <= route_exploration <= 1.0:
            raise TrainingError("`route_exploration` debe estar entre cero y uno.")
        if route_callback is None:
            route_callback = RouteConvergenceCallback()
        elif not callable(route_callback):
            raise TrainingError("`route_callback` debe ser invocable.")

        target_layers = {sample["target_layer"] for sample in dataset}
        if len(target_layers) != 1:
            raise DatasetError("Todos los ejemplos deben compartir target_layer.")
        output_layer = target_layers.pop()
        if output_layer < 0 or output_layer >= len(self.network.layers):
            raise ArchitectureError(f"target_layer fuera de rango: {output_layer}.")
        output_size = self.network.layers[output_layer].num_neurons
        for sample in dataset:
            target = sample["target_neuron"]
            if target < 0 or target >= output_size:
                raise DatasetError(f"target_neuron fuera de rango: {target}.")

        self._configure_connectivity(lineal)
        readout = [
            synapse for _, post_idx, synapse in self.network.connections
            if post_idx == output_layer
            and (
                hasattr(synapse, "apply_supervised_update")
                or hasattr(synapse, "W")
            )
        ]
        if not readout:
            raise ArchitectureError(
                "La capa objetivo necesita al menos una sinapsis entrante con pesos."
            )

        history = {
            "loss": [],
            "accuracy": [],
            "mean_steps": [],
            "mean_route_attention": [],
            "route_history": [],
            "stopped_early": False,
            "lineal": lineal,
        }
        if validation_data is not None:
            history["val_accuracy"] = []

        frame_index = 0
        for epoch in range(epochs):
            correct = 0
            losses = []
            steps_taken = 0
            route_attention_total = 0.0
            route_uses = {}
            route_steps = {}
            for sample_index, sample in enumerate(dataset):
                inputs = {
                    layer_idx: np.asarray(values, dtype=float)
                    for layer_idx, values in sample["inputs"].items()
                }
                target = sample["target_neuron"]
                route = (
                    self.network.select_route(
                        output_layer,
                        exploration_rate=route_exploration,
                        source_layers=inputs,
                    )
                    if adaptive_routing
                    else None
                )
                active_readout = [
                    (index, synapse)
                    for index, (_, post_idx, synapse) in enumerate(
                        self.network.connections
                    )
                    if post_idx == output_layer
                    and (
                        hasattr(synapse, "apply_supervised_update")
                        or hasattr(synapse, "W")
                    )
                    and (route is None or index in route)
                ]
                if not active_readout:
                    raise ArchitectureError(
                        "La ruta seleccionada no contiene sinapsis entrenables "
                        "hacia la capa objetivo."
                    )
                self.network.reset_traces(reset_neuromodulators=True)
                output_spikes = np.zeros(output_size, dtype=float)
                stable_target_steps = 0
                active_layers = {
                    layer_idx
                    for index in (
                        range(len(self.network.connections))
                        if route is None
                        else route
                    )
                    for layer_idx in self.network.connections[index][:2]
                }
                layer_activity = {
                    layer_idx: np.zeros(
                        self.network.layers[layer_idx].num_neurons,
                        dtype=float,
                    )
                    for layer_idx in active_layers
                }
                presynaptic_spikes = {
                    index: np.zeros(synapse.pre_size, dtype=float)
                    for index, synapse in active_readout
                }

                for step_index in range(steps_per_sample):
                    spikes = self.network.step(
                        inputs,
                        learn=False,
                        active_connections=route,
                        **sample.get("chems", {}),
                    )
                    output_spikes += spikes[output_layer]
                    steps_taken += 1
                    for layer_idx in active_layers:
                        layer = self.network.layers[layer_idx]
                        threshold_range = max(
                            abs(layer.base_v_thresh - layer.v_rest),
                            1e-12,
                        )
                        voltage_activity = np.clip(
                            (layer.v - layer.v_rest) / threshold_range,
                            0.0,
                            1.0,
                        )
                        layer_activity[layer_idx] += np.maximum(
                            voltage_activity,
                            spikes.get(
                                layer_idx,
                                np.zeros(layer.num_neurons),
                            ),
                        )
                    for index, synapse in active_readout:
                        pre_idx = self.network.connections[index][0]
                        presynaptic_spikes[index] += spikes[pre_idx]
                    if activity_callback is not None:
                        activity_callback(
                            {
                                "frame": frame_index,
                                "epoch": epoch + 1,
                                "sample": sample_index,
                                "step": step_index,
                                "active_layers": sorted(
                                    self.network.last_active_layers
                                ),
                                "active_connections": sorted(
                                    self.network.last_active_connections
                                ),
                                "spikes": {
                                    str(layer_idx): np.flatnonzero(layer_spikes).tolist()
                                    for layer_idx, layer_spikes in spikes.items()
                                    if np.any(layer_spikes)
                                },
                            }
                        )
                    frame_index += 1
                    is_target_winning = (
                        output_spikes[target] > 0
                        and np.count_nonzero(
                            output_spikes == output_spikes[target]
                        ) == 1
                    )
                    stable_target_steps = (
                        stable_target_steps + 1 if is_target_winning else 0
                    )
                    if (
                        step_index + 1 >= min(3, steps_per_sample)
                        and stable_target_steps >= 2
                    ):
                        break

                prediction = (
                    int(np.argmax(output_spikes))
                    if np.any(output_spikes)
                    else -1
                )
                correct += prediction == target

                sample_steps = step_index + 1
                route_attention_total += self.network.last_route_attention
                if route is not None:
                    route_key = tuple(sorted(route))
                    route_uses[route_key] = route_uses.get(route_key, 0) + 1
                    route_steps[route_key] = (
                        route_steps.get(route_key, 0) + sample_steps
                    )
                rates = output_spikes / sample_steps
                logits = rates * 4.0
                probabilities = np.exp(logits - np.max(logits))
                probabilities /= probabilities.sum()
                losses.append(-np.log(max(probabilities[target], 1e-12)))
                error = -probabilities
                error[target] += 1.0
                excess_gaba = max(
                    0.0,
                    self.network.neuromod.GABA
                    - self.network.neuromod.gaba_baseline,
                ) / max(
                    1.0 - self.network.neuromod.gaba_baseline,
                    1e-12,
                )
                stress_plasticity_gate = max(0.25, 1.0 - 0.75 * excess_gaba)
                focused_learning_rate = learning_rate * (
                    1.0 + self.network.last_route_attention
                ) * stress_plasticity_gate

                if lineal:
                    self._apply_linear_supervision(
                        route,
                        error,
                        output_layer,
                        layer_activity,
                        sample_steps,
                        focused_learning_rate,
                    )
                else:
                    for index, synapse in active_readout:
                        pre_rate = presynaptic_spikes[index] / sample_steps
                        if not np.any(pre_rate):
                            continue
                        if hasattr(synapse, "apply_supervised_update"):
                            synapse.apply_supervised_update(
                                error,
                                pre_rate,
                                focused_learning_rate,
                            )
                            continue
                        mask = getattr(synapse, "connectivity_mask", 1.0)
                        synapse.W += (
                            focused_learning_rate
                            * np.outer(error, pre_rate)
                            * mask
                        )
                        synapse.W[:] = np.clip(
                            synapse.W, 0.0, 1.2
                        ) * mask
                        row_sums = np.sum(
                            synapse.W, axis=1, keepdims=True
                        )
                        max_row_sum = getattr(
                            synapse, "max_row_sum", np.inf
                        )
                        scale = np.minimum(
                            1.0,
                            max_row_sum / np.maximum(row_sums, 1e-12),
                        )
                        synapse.W *= scale
                updated_post_layers = {
                    self.network.connections[index][1]
                    for index in (
                        range(len(self.network.connections))
                        if route is None
                        else route
                    )
                }
                self.network.enforce_dendrite_limits(updated_post_layers)
                if adaptive_routing:
                    self.network.update_route_scores(
                        route,
                        1.0 if prediction == target else -1.0,
                        learning_rate=0.25,
                    )

            history["loss"].append(float(np.mean(losses)))
            history["accuracy"].append(correct / len(dataset))
            history["mean_steps"].append(steps_taken / len(dataset))
            history["mean_route_attention"].append(
                route_attention_total / len(dataset)
            )
            should_stop = False
            if adaptive_routing:
                route_summary = {
                    "epoch": epoch + 1,
                    "mean_steps": history["mean_steps"][-1],
                    "mean_route_attention": history["mean_route_attention"][-1],
                    "routes": [
                        {
                            "route": list(route_key),
                            "uses": uses,
                            "mean_steps": route_steps[route_key] / uses,
                        }
                        for route_key, uses in sorted(route_uses.items())
                    ],
                }
                should_stop = bool(route_callback(route_summary))
                if lineal:
                    should_stop = False
                history["route_history"].append(route_summary)
                if hasattr(route_callback, "best_route"):
                    history["best_route"] = list(route_callback.best_route)
                    history["best_route_uses"] = route_callback.best_route_uses
                    history["best_route_mean_steps"] = (
                        route_callback.best_route_mean_steps
                    )
                    history["best_response_steps"] = (
                        route_callback.best_response_steps
                    )
            if validation_data is not None:
                metrics = self.evaluate(
                    validation_data,
                    steps_per_sample,
                    adaptive_routing=adaptive_routing,
                )
                history["val_accuracy"].append(metrics["accuracy"])
            if should_stop:
                history["stopped_early"] = True
                break

        self.network.reset_traces(reset_neuromodulators=True)
        return history

    def fit_dataset(
        self,
        dataset,
        target_layer=None,
        epochs=10,
        steps_per_sample=30,
        learning_rate=0.2,
        validation_data=None,
        adaptive_routing=True,
        route_exploration=0.15,
        activity_callback=None,
        route_callback=None,
        lineal=True,
    ):
        """Entrena desde un SpikeDataset sin construir muestras internas a mano."""
        if not hasattr(dataset, "features") or not hasattr(dataset, "targets"):
            raise DatasetError("`dataset` debe ser un SpikeDataset etiquetado.")
        if len(dataset.features) != len(dataset.targets) or not len(dataset.targets):
            raise DatasetError("El dataset debe contener entradas y etiquetas.")
        if validation_data is not None and (
            not hasattr(validation_data, "features")
            or not hasattr(validation_data, "targets")
        ):
            raise DatasetError("`validation_data` debe ser un SpikeDataset etiquetado.")
        if target_layer is None:
            if not self.network.layers:
                raise ArchitectureError("La red no tiene una capa de salida.")
            target_layer = len(self.network.layers) - 1

        def make_samples(data):
            if len(data.features) != len(data.targets):
                raise DatasetError("Entradas y etiquetas de un dataset no coinciden.")
            return [
                {
                    "inputs": {0: features},
                    "target_layer": target_layer,
                    "target_neuron": int(target),
                }
                for features, target in zip(data.features, data.targets)
            ]

        samples = make_samples(dataset)
        validation_samples = (
            make_samples(validation_data)
            if validation_data is not None
            else None
        )
        return self.fit_supervised(
            samples,
            epochs=epochs,
            steps_per_sample=steps_per_sample,
            learning_rate=learning_rate,
            validation_data=validation_samples,
            adaptive_routing=adaptive_routing,
            route_exploration=route_exploration,
            activity_callback=activity_callback,
            route_callback=route_callback,
            lineal=lineal,
        )

    def evaluate(self, dataset, steps_per_sample=30, adaptive_routing=True):
        if not dataset:
            raise DatasetError("El dataset de evaluación está vacío.")
        if steps_per_sample <= 0:
            raise TrainingError("steps_per_sample debe ser mayor que cero.")

        predictions = []
        targets = []
        spike_counts = []
        for sample in dataset:
            output_layer = sample["target_layer"]
            route = (
                self.network.select_route(
                    output_layer,
                    source_layers=sample["inputs"],
                )
                if adaptive_routing
                else None
            )
            prediction, counts = self.network.predict(
                sample["inputs"],
                output_layer_idx=output_layer,
                steps=steps_per_sample,
                active_connections=route,
            )
            predictions.append(prediction)
            targets.append(sample["target_neuron"])
            spike_counts.append(counts)

        return {
            "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(targets))),
            "predictions": np.asarray(predictions, dtype=int),
            "targets": np.asarray(targets, dtype=int),
            "spike_counts": np.asarray(spike_counts, dtype=int),
        }
