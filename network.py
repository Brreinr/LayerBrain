import json
import os
import tempfile
import zipfile

import numpy as np

from .errors import (
    ArchitectureError,
    IncompatibleDimensionsError,
    PersistenceError,
)
from .neuromodulation import NeuromodulatorySystem

class BrainNetwork:
    def __init__(self, neuromodulation=None):
        self.layers = []
        self.connections = []
        self._incoming_connections = {}
        self.route_scores = []
        self.lineal_mode = None
        self.last_active_layers = frozenset()
        self.last_active_connections = frozenset()
        self.last_processed_synapses = 0
        self.last_skipped_synapses = 0
        self.last_route_attention = 0.0
        self.last_spikes = {}
        self.neuromod = (
            NeuromodulatorySystem()
            if neuromodulation is None
            else neuromodulation
        )
        self.prev_inputs = {}

    def add_layer(self, layer):
        self.layers.append(layer)
        return len(self.layers) - 1
    
    def connect(self, pre_idx, post_idx, synapse):
        if not isinstance(pre_idx, (int, np.integer)) or not 0 <= pre_idx < len(self.layers):
            raise ArchitectureError(f"Índice de capa de origen inválido: {pre_idx}.")
        if not isinstance(post_idx, (int, np.integer)) or not 0 <= post_idx < len(self.layers):
            raise ArchitectureError(f"Índice de capa de destino inválido: {post_idx}.")
        if not all(hasattr(synapse, name) for name in ("pre_size", "post_size", "step")):
            raise ArchitectureError(
                "La sinapsis debe definir `pre_size`, `post_size` y el método `step`."
            )
        pre_layer = self.layers[pre_idx]
        post_layer = self.layers[post_idx]
        if synapse.pre_size != pre_layer.num_neurons or synapse.post_size != post_layer.num_neurons:
            raise IncompatibleDimensionsError(
                f"Dimensiones incompatibles en la conexión: "
                f"La sinapsis requiere pre_size={synapse.pre_size} (capa tiene {pre_layer.num_neurons}) "
                f"y post_size={synapse.post_size} (capa destino tiene {post_layer.num_neurons})."
            )
        if hasattr(synapse, "set_num_dendrites"):
            synapse.set_num_dendrites(post_layer.num_dendrites)
        self.connections.append((pre_idx, post_idx, synapse))
        self.route_scores.append(0.0)
        self._incoming_connections.setdefault(post_idx, []).append(
            len(self.connections) - 1
        )
        self._enforce_dendrite_limit(post_idx)

    def _enforce_dendrite_limit(self, post_idx):
        limit = self.layers[post_idx].num_dendrites
        incoming = [
            (index, synapse)
            for index, (_, target, synapse) in enumerate(self.connections)
            if target == post_idx
        ]
        sparse_keep = {
            index: np.ones(synapse.connection_count, dtype=bool)
            for index, synapse in incoming
            if hasattr(synapse, "pre_indices")
        }
        for neuron_idx in range(self.layers[post_idx].num_neurons):
            edges = []
            for index, synapse in incoming:
                if hasattr(synapse, "post_indices"):
                    start, end = synapse.post_offsets[neuron_idx : neuron_idx + 2]
                    edges.extend(
                        (index, int(edge_idx))
                        for edge_idx in synapse.post_edge_indices[start:end]
                    )
                elif hasattr(synapse, "W"):
                    edges.extend(
                        (index, int(pre_idx))
                        for pre_idx in np.flatnonzero(synapse.W[neuron_idx])
                    )
            if len(edges) <= limit:
                continue
            keep_indices = set(
                np.random.choice(len(edges), size=limit, replace=False).tolist()
            )
            for edge_position, (index, edge_idx) in enumerate(edges):
                if edge_position in keep_indices:
                    continue
                synapse = self.connections[index][2]
                if hasattr(synapse, "post_indices"):
                    sparse_keep[index][edge_idx] = False
                else:
                    synapse.W[neuron_idx, edge_idx] = 0.0
                    if hasattr(synapse, "connectivity_mask"):
                        synapse.connectivity_mask[neuron_idx, edge_idx] = 0.0
        for index, keep in sparse_keep.items():
            synapse = self.connections[index][2]
            if np.all(keep):
                continue
            synapse.pre_indices = synapse.pre_indices[keep]
            synapse.post_indices = synapse.post_indices[keep]
            synapse.weights = synapse.weights[keep]
            synapse.eligibility = synapse.eligibility[keep]
            synapse._rebuild_indices()

    def enforce_dendrite_limits(self, post_layers=None):
        """Applies each destination neuron's dendrite limit across all inputs."""
        targets = (
            range(len(self.layers))
            if post_layers is None
            else set(post_layers)
        )
        for post_idx in targets:
            self._enforce_dendrite_limit(post_idx)

    def select_route(
        self,
        target_layer,
        exploration_rate=0.0,
        source_layers=None,
    ):
        """Elige una ruta hacia la salida, favoreciendo conexiones útiles."""
        if not isinstance(target_layer, (int, np.integer)) or not 0 <= target_layer < len(self.layers):
            raise ArchitectureError(f"Índice de capa objetivo inválido: {target_layer}.")
        if not np.isfinite(exploration_rate) or not 0.0 <= exploration_rate <= 1.0:
            raise ArchitectureError("`exploration_rate` debe estar entre cero y uno.")

        route = set()
        visited = set()
        source_layers = (
            frozenset()
            if source_layers is None
            else frozenset(source_layers)
        )
        if any(
            not isinstance(index, (int, np.integer))
            or not 0 <= index < len(self.layers)
            for index in source_layers
        ):
            raise ArchitectureError("`source_layers` contiene índices inválidos.")

        def grow_path(layer_idx):
            if (
                layer_idx in source_layers
                or not self._incoming_connections.get(layer_idx)
                or layer_idx in visited
            ):
                return
            visited.add(layer_idx)
            incoming = [
                index
                for index in self._incoming_connections.get(layer_idx, ())
                if self.connections[index][0] not in visited
            ]
            if not incoming:
                return
            scores = np.asarray([self.route_scores[index] for index in incoming])
            if np.random.random() < exploration_rate:
                chosen = int(np.random.choice(incoming))
            else:
                best_score = np.max(scores)
                best_connections = [
                    index
                    for index, score in zip(incoming, scores)
                    if score == best_score
                ]
                chosen = int(np.random.choice(best_connections))
            route.add(chosen)
            grow_path(self.connections[chosen][0])

        grow_path(int(target_layer))
        if not route:
            raise ArchitectureError(
                f"No existe una ruta de conexiones hacia la capa {target_layer}."
            )
        return frozenset(route)

    def update_route_scores(self, route, reward, learning_rate=0.1):
        """Actualiza la preferencia de las conexiones que formaron una ruta."""
        if not np.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ArchitectureError("`reward` de ruta debe estar entre -1 y uno.")
        if not np.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
            raise ArchitectureError("`learning_rate` de ruta debe estar en (0, 1].")
        for index in route:
            self.route_scores[index] = (
                (1.0 - learning_rate) * self.route_scores[index]
                + learning_rate * reward
            )

    @property
    def memory_usage_bytes(self):
        """Bytes currently held by NumPy arrays owned by this network."""
        arrays = [np.asarray(self.route_scores)]
        for layer in self.layers:
            arrays.extend(
                value for value in vars(layer).values() if isinstance(value, np.ndarray)
            )
        for _, _, synapse in self.connections:
            arrays.extend(
                value
                for value in vars(synapse).values()
                if isinstance(value, np.ndarray)
            )
        unique_arrays = {id(array): array for array in arrays}
        return sum(array.nbytes for array in unique_arrays.values())

    def step(
        self,
        external_inputs,
        novelty=None,
        reward=0.0,
        fatigue=None,
        uncertainty=0.0,
        social=None,
        dt=1.0,
        learn=True,
        active_connections=None,
    ):
        # 1. NOVEDAD AUTOMÁTICA -> Acetilcolina (ACh)
        auto_novelty = 0.0
        clean_inputs = {}
        modality_changes = []
        
        if external_inputs:
            diffs = []
            for idx, current_val in external_inputs.items():
                if callable(current_val):
                    continue
                if not isinstance(idx, (int, np.integer)) or not 0 <= idx < len(self.layers):
                    raise ArchitectureError(f"Índice de capa de entrada inválido: {idx}.")
                try:
                    arr_curr = np.asarray(current_val, dtype=float).flatten()
                except (TypeError, ValueError) as exc:
                    raise IncompatibleDimensionsError(
                        f"La entrada para la capa {idx} debe ser numérica."
                    ) from exc
                expected_size = self.layers[idx].num_neurons
                if arr_curr.shape != (expected_size,):
                    raise IncompatibleDimensionsError(
                        f"La entrada para la capa {idx} tiene tamaño {arr_curr.size}; "
                        f"se esperaban {expected_size} valores."
                    )
                if not np.all(np.isfinite(arr_curr)):
                    raise IncompatibleDimensionsError(
                        f"La entrada para la capa {idx} contiene NaN o infinito."
                    )
                clean_inputs[idx] = arr_curr
                
                if idx in self.prev_inputs:
                    arr_prev = self.prev_inputs[idx]
                    if arr_curr.shape == arr_prev.shape:
                        diffs.append(float(np.mean(np.abs(arr_curr - arr_prev))))
                        scale = max(
                            float(np.mean(np.abs(arr_curr))),
                            float(np.mean(np.abs(arr_prev))),
                            1e-12,
                        )
                        modality_changes.append(
                            float(
                                np.clip(
                                    np.mean(np.abs(arr_curr - arr_prev)) / scale,
                                    0.0,
                                    1.0,
                                )
                            )
                        )
            if diffs:
                auto_novelty = float(np.mean(diffs)) / 80.0

        self.prev_inputs = {k: np.array(v, dtype=float) for k, v in clean_inputs.items()}

        # 2. SOBRECARGA DE DISPAROS -> GABA
        auto_overload = 0.0
        if self.last_spikes:
            total_spikes = sum(float(np.sum(s)) for s in self.last_spikes.values())
            active_neurons = sum(
                self.layers[index].num_neurons
                for index in self.last_spikes
            )
            if active_neurons > 0:
                spike_rate = total_spikes / active_neurons
                auto_overload = float(
                    np.clip(
                        (spike_rate - self.neuromod.GABA_OVERLOAD_THRESHOLD)
                        / (1.0 - self.neuromod.GABA_OVERLOAD_THRESHOLD),
                        0.0,
                        1.0,
                    )
                )

        # 3. COHERENCIA MULTISENSORIAL -> Oxitocina (OXT)
        auto_social = 0.0
        if len(clean_inputs) >= 2:
            active_densities = []
            for values in clean_inputs.values():
                peak = float(np.max(np.abs(values)))
                if peak <= 0.0:
                    continue
                active_ratio = np.count_nonzero(np.abs(values) >= peak * 0.1)
                active_densities.append(active_ratio / values.size)
            if len(active_densities) >= 2:
                density_disagreement = max(active_densities) - min(active_densities)
                coherence = float(np.exp(-2.0 * density_disagreement))
                familiarity = (
                    float(np.exp(-np.mean(modality_changes)))
                    if modality_changes
                    else 0.5
                )
                auto_social = coherence * (0.5 + 0.5 * familiarity)

        eff_novelty = auto_novelty if novelty is None else float(novelty)
        eff_fatigue = 0.0 if fatigue is None else float(fatigue)
        eff_social = auto_social if social is None else float(social)

        self.neuromod.step(
            novelty=eff_novelty,
            reward=reward,
            fatigue=eff_fatigue,
            overload=auto_overload,
            uncertainty=uncertainty,
            social=eff_social,
            dt=dt,
        )
        
        if active_connections is None:
            connection_indices = range(len(self.connections))
            active_layers = set(range(len(self.layers)))
        else:
            connection_indices = tuple(active_connections)
            if any(
                not isinstance(index, (int, np.integer))
                or not 0 <= index < len(self.connections)
                for index in connection_indices
            ):
                raise ArchitectureError("`active_connections` contiene índices inválidos.")
            active_layers = set()
            for index in connection_indices:
                pre_idx, post_idx, _ = self.connections[index]
                active_layers.update((pre_idx, post_idx))
            if not connection_indices:
                active_layers.update(clean_inputs)
            else:
                active_layers.update(
                    index for index in clean_inputs if index in active_layers
                )
        self.last_active_layers = frozenset(active_layers)
        self.last_active_connections = frozenset(connection_indices)
        if active_connections is None or not connection_indices:
            self.last_route_attention = 0.0
        else:
            route_scores = np.asarray(
                [self.route_scores[index] for index in connection_indices],
                dtype=float,
            )
            self.last_route_attention = float(
                np.mean(np.clip(route_scores, 0.0, 1.0))
            )
        self.neuromod.set_route_attention(self.last_route_attention)

        currents = {
            i: np.zeros(self.layers[i].num_neurons)
            for i in active_layers
        }
        processed_synapses = 0
        skipped_synapses = 0
        for idx, I_ext in clean_inputs.items():
            if idx in currents:
                currents[idx] += I_ext

        for index in connection_indices:
            pre_idx, post_idx, synapse = self.connections[index]
            pre_spikes = self.last_spikes.get(pre_idx)
            if pre_spikes is None:
                pre_spikes = np.zeros(synapse.pre_size)
            post_spikes = self.last_spikes.get(post_idx, None)
            has_presynaptic_event = bool(np.any(pre_spikes))
            has_pending_current = (
                bool(synapse.has_pending_current)
                if hasattr(synapse, "has_pending_current")
                else hasattr(synapse, "g") and bool(np.any(synapse.g))
            )
            if (
                not learn
                and not has_presynaptic_event
                and not has_pending_current
            ):
                skipped_synapses += 1
                continue
            I_syn = synapse.step(pre_spikes, post_spikes, neuromod=self.neuromod, dt=dt, learn=learn)
            currents[post_idx] += I_syn
            processed_synapses += 1

        self.last_processed_synapses = processed_synapses
        self.last_skipped_synapses = skipped_synapses

        self.last_spikes = {}
        for i, layer in enumerate(self.layers):
            if i in active_layers:
                self.last_spikes[i] = layer.step(
                    currents[i], neuromod=self.neuromod, dt=dt
                )

        return self.last_spikes
    
    def reset_traces(self, reset_neuromodulators=False):
        self.prev_inputs = {}
        self.last_spikes = {}
        self.last_processed_synapses = 0
        self.last_skipped_synapses = 0
        self.last_route_attention = 0.0
        for layer in self.layers:
            layer.v.fill(layer.v_rest)
            layer.refractory_time.fill(0.0)
            if hasattr(layer, 'w_adapt'):
                layer.w_adapt.fill(0.0)
            if hasattr(layer, 'sensory_adaptation'):
                layer.sensory_adaptation.fill(0.0)
        for _, _, synapse in self.connections:
            synapse.g.fill(0.0)
            if hasattr(synapse, "_g_scale"):
                synapse._g_scale = 1.0
            if hasattr(synapse, "_eligibility_scale"):
                synapse._eligibility_scale = 1.0
            if hasattr(synapse, "_trace_pre_scale"):
                synapse._trace_pre_scale = 1.0
            if hasattr(synapse, "_trace_post_scale"):
                synapse._trace_post_scale = 1.0
            if hasattr(synapse, "E"):
                synapse.E.fill(0.0)
            if hasattr(synapse, "eligibility"):
                synapse.eligibility.fill(0.0)
            if hasattr(synapse, 'trace_pre'):
                synapse.trace_pre.fill(0.0)
            if hasattr(synapse, 'trace_post'):
                synapse.trace_post.fill(0.0)
        if reset_neuromodulators:
            self.neuromod.reset()

    def quantize_for_inference(self):
        """Comprime pesos dispersos a enteros fijos de 16 bits para inferencia."""
        for _, _, synapse in self.connections:
            if hasattr(synapse, "quantize_weights"):
                synapse.quantize_weights()

    def dequantize_weights(self):
        """Restaura pesos dispersos float32 para edición o entrenamiento."""
        for _, _, synapse in self.connections:
            if hasattr(synapse, "dequantize_weights"):
                synapse.dequantize_weights()

    def predict(
        self,
        external_inputs: dict,
        output_layer_idx: int,
        steps: int = 25,
        dt: float = 1.0,
        active_connections=None,
        early_exit: bool = True,
    ):
        """
        Ejecuta el cerebro en modo inferencia rápida y devuelve la acción ganadora
        junto con el conteo de spikes de la capa de salida.
        """
        if steps <= 0:
            raise ValueError("steps debe ser mayor que cero.")
        if output_layer_idx < 0 or output_layer_idx >= len(self.layers):
            raise IndexError("output_layer_idx está fuera del rango de capas.")
        if not isinstance(early_exit, bool):
            raise ValueError("early_exit debe ser booleano.")
        self.reset_traces(reset_neuromodulators=True)
        total_output_spikes = np.zeros(self.layers[output_layer_idx].num_neurons, dtype=int)
        previous_winner = -1
        stable_winner_steps = 0

        for step_index in range(steps):
            if total_output_spikes.size > 1 and np.any(total_output_spikes):
                probabilities = total_output_spikes / np.sum(total_output_spikes)
                entropy = -np.sum(
                    probabilities[probabilities > 0]
                    * np.log(probabilities[probabilities > 0])
                )
                uncertainty = float(
                    np.clip(
                        entropy / np.log(total_output_spikes.size),
                        0.0,
                        1.0,
                    )
                )
            else:
                uncertainty = 1.0
            spikes = self.step(
                external_inputs,
                reward=0.0,
                uncertainty=uncertainty,
                dt=dt,
                learn=False,
                active_connections=active_connections,
            )
            total_output_spikes += spikes[output_layer_idx]

            if early_exit and step_index + 1 >= 3 and np.any(total_output_spikes):
                max_spikes = np.max(total_output_spikes)
                winners = np.flatnonzero(total_output_spikes == max_spikes)
                winner = int(winners[0]) if winners.size == 1 else -1
                if winner >= 0 and winner == previous_winner:
                    stable_winner_steps += 1
                else:
                    stable_winner_steps = 1 if winner >= 0 else 0
                previous_winner = winner
                if stable_winner_steps >= 3:
                    break

        winner_action = (
            int(np.argmax(total_output_spikes))
            if np.any(total_output_spikes)
            else (
                int(np.argmax(self.layers[output_layer_idx].v))
                if self.lineal_mode is True
                else -1
            )
        )
        return winner_action, total_output_spikes

    def _persistence_metadata(self, model_metadata=None):
        return {
            "format_version": 1,
            "lineal_mode": self.lineal_mode,
            "neuromodulation": {
                name: self._json_safe(getattr(self.neuromod, name))
                for name in (
                    "dt",
                    "tau_ACh",
                    "tau_DA",
                    "tau_GABA",
                    "tau_OXT",
                    "gaba_baseline",
                    "route_attention_gain",
                )
            },
            "layers": [
                {
                    "type": type(layer).__name__,
                    "num_neurons": int(layer.num_neurons),
                    "num_dendrites": int(layer.num_dendrites),
                    "parameters": self._json_safe(
                        {
                            name: value
                            for name, value in vars(layer).items()
                            if not name.startswith("_")
                            and not isinstance(value, np.ndarray)
                        }
                    ),
                }
                for layer in self.layers
            ],
            "connections": [
                {
                    "pre": int(pre_idx),
                    "post": int(post_idx),
                    "type": type(synapse).__name__,
                    "pre_size": int(synapse.pre_size),
                    "post_size": int(synapse.post_size),
                    "parameters": self._json_safe(
                        {
                            name: value
                            for name, value in vars(synapse).items()
                            if not name.startswith("_")
                            and not isinstance(value, np.ndarray)
                        }
                    ),
                }
                for pre_idx, post_idx, synapse in self.connections
            ],
            "model": model_metadata,
        }

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {
                str(key): BrainNetwork._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [BrainNetwork._json_safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Valor no serializable en metadatos: {type(value).__name__}.")

    @staticmethod
    def _decode_persistence_metadata(data, filepath):
        if "__metadata__" not in data:
            return None
        try:
            encoded = np.asarray(data["__metadata__"])
            if encoded.dtype != np.uint8 or encoded.ndim != 1:
                raise ValueError("metadatos con formato inválido")
            metadata = json.loads(encoded.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise PersistenceError(
                f"Los metadatos del modelo {filepath!r} están dañados."
            ) from exc
        if (
            not isinstance(metadata, dict)
            or metadata.get("format_version") != 1
            or not isinstance(metadata.get("layers"), list)
            or not isinstance(metadata.get("connections"), list)
        ):
            raise PersistenceError(
                f"El formato del modelo {filepath!r} no es compatible."
            )
        return metadata

    def read_brain_metadata(self, filepath):
        """Lee y valida los metadatos sin modificar la red."""
        try:
            with np.load(filepath, allow_pickle=False) as data:
                return self._decode_persistence_metadata(data, filepath)
        except PersistenceError:
            raise
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise PersistenceError(
                f"No se pudo abrir el modelo {filepath!r}; comprueba la ruta y el archivo."
            ) from exc

    def save_brain(
        self,
        filepath: str = "layerbrain_weights.npz",
        model_metadata=None,
    ):
        """Guarda pesos, topología y metadatos en un archivo comprimido."""
        state_dict = {}
        for idx, (_, _, syn) in enumerate(self.connections):
            if hasattr(syn, "state_dict"):
                sparse_state = syn.state_dict()
                for name, values in sparse_state.items():
                    state_dict[f"syn_{idx}_{name}"] = values
            elif hasattr(syn, 'W'):
                state_dict[f"syn_{idx}_W"] = syn.W.astype(np.float16)  # Comprimido a 16 bits
            if hasattr(syn, 'connectivity_mask'):
                state_dict[f"syn_{idx}_mask"] = syn.connectivity_mask.astype(np.uint8)
        state_dict["route_scores"] = np.asarray(self.route_scores, dtype=np.float64)
        metadata = self._persistence_metadata(model_metadata)
        try:
            state_dict["__metadata__"] = np.frombuffer(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8"),
                dtype=np.uint8,
            )
        except (TypeError, ValueError) as exc:
            raise PersistenceError(
                "No se pudieron serializar los metadatos de la arquitectura."
            ) from exc
        filepath = os.fspath(filepath)
        temporary_path = None
        try:
            directory = os.path.dirname(os.path.abspath(filepath))
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=".layerbrain-",
                suffix=".tmp",
                dir=directory,
                delete=False,
            ) as temporary_file:
                temporary_path = temporary_file.name
                np.savez_compressed(temporary_file, **state_dict)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, filepath)
        except OSError as exc:
            raise PersistenceError(f"No se pudo guardar el conectoma en {filepath!r}.") from exc
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
        return filepath

    def load_brain(self, filepath: str = "layerbrain_weights.npz", model_metadata=None):
        """Carga pesos solo si el archivo coincide con la arquitectura actual."""
        try:
            data = np.load(filepath, allow_pickle=False)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise PersistenceError(
                f"No se pudo abrir el modelo {filepath!r}; comprueba la ruta y el archivo."
            ) from exc

        weights_to_load = []
        incoming_counts = [
            np.zeros(layer.num_neurons, dtype=np.int64)
            for layer in self.layers
        ]
        try:
            with data:
                metadata = self._decode_persistence_metadata(data, filepath)
                if metadata is not None:
                    expected_metadata = self._persistence_metadata(model_metadata)
                    if metadata != expected_metadata:
                        raise PersistenceError(
                            "El archivo guardado pertenece a otra arquitectura, "
                            "topología o configuración de clases."
                        )
                for idx, (_, _, syn) in enumerate(self.connections):
                    weight_key = f"syn_{idx}_W"
                    mask_key = f"syn_{idx}_mask"
                    if hasattr(syn, "load_state"):
                        keys = (
                            f"syn_{idx}_pre_indices",
                            f"syn_{idx}_post_indices",
                            f"syn_{idx}_weights",
                        )
                        if any(key not in data for key in keys):
                            raise PersistenceError(
                                f"El modelo no contiene el estado disperso completo "
                                f"para la conexión {idx}."
                            )
                        sparse_state = tuple(np.asarray(data[key]) for key in keys)
                        if (
                            sparse_state[0].ndim != 1
                            or sparse_state[0].dtype.kind not in "iu"
                            or sparse_state[1].shape != sparse_state[0].shape
                            or sparse_state[1].dtype.kind not in "iu"
                            or sparse_state[2].shape != sparse_state[0].shape
                            or not np.all(np.isfinite(sparse_state[2]))
                            or np.any(sparse_state[0] < 0)
                            or np.any(sparse_state[0] >= syn.pre_size)
                            or np.any(sparse_state[1] < 0)
                            or np.any(sparse_state[1] >= syn.post_size)
                            or np.unique(
                                np.column_stack(sparse_state[:2]), axis=0
                            ).shape[0]
                            != sparse_state[0].size
                        ):
                            raise PersistenceError(
                                f"El estado disperso de la conexión {idx} es inválido."
                            )
                        post_idx = self.connections[idx][1]
                        incoming_counts[post_idx] += np.bincount(
                            sparse_state[1],
                            minlength=self.layers[post_idx].num_neurons,
                        )
                        weights_to_load.append((syn, "sparse", sparse_state))
                    elif hasattr(syn, 'W'):
                        if weight_key not in data:
                            raise PersistenceError(
                                f"El modelo no contiene los pesos requeridos {weight_key!r}."
                            )
                        weights = data[weight_key]
                        if weights.shape != syn.W.shape or not np.all(np.isfinite(weights)):
                            raise PersistenceError(
                                f"Los pesos {weight_key!r} tienen forma o valores incompatibles."
                            )
                        post_idx = self.connections[idx][1]
                        incoming_counts[post_idx] += np.count_nonzero(
                            weights, axis=1
                        )
                        weights_to_load.append((syn, "W", weights.astype(syn.W.dtype)))
                    if hasattr(syn, 'connectivity_mask'):
                        if mask_key not in data:
                            raise PersistenceError(
                                f"El modelo no contiene la máscara requerida {mask_key!r}."
                            )
                        mask = data[mask_key]
                        if mask.shape != syn.connectivity_mask.shape or not np.all(np.isfinite(mask)):
                            raise PersistenceError(
                                f"La máscara {mask_key!r} tiene forma o valores incompatibles."
                            )
                        weights_to_load.append(
                            (syn, "connectivity_mask", mask.astype(syn.connectivity_mask.dtype))
                        )
                if any(
                    np.any(counts > layer.num_dendrites)
                    for counts, layer in zip(incoming_counts, self.layers)
                ):
                    raise PersistenceError(
                        "La conectividad guardada supera el límite de dendritas "
                        "de una o más neuronas."
                    )
                if "route_scores" in data:
                    route_scores = data["route_scores"]
                    if route_scores.shape != (len(self.connections),) or not np.all(np.isfinite(route_scores)):
                        raise PersistenceError("Las puntuaciones de enrutamiento tienen forma o valores inválidos.")
                    weights_to_load.append(
                        (self, "route_scores", route_scores.astype(float).tolist())
                    )
        except PersistenceError:
            raise
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise PersistenceError(
                f"No se pudo leer el modelo {filepath!r}; el archivo está dañado o incompleto."
            ) from exc

        for syn, attribute, value in weights_to_load:
            if attribute == "sparse":
                try:
                    syn.load_state(*value)
                except ValueError as exc:
                    raise PersistenceError(
                        f"No se pudo cargar el estado disperso: {exc}"
                    ) from exc
            else:
                setattr(syn, attribute, value)
        self.reset_traces()
        return self
