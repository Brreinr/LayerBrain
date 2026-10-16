import numpy as np


def _validate_sizes(pre_size, post_size):
    if pre_size <= 0 or post_size <= 0:
        raise ValueError("pre_size y post_size deben ser mayores que cero.")


def _validate_dynamics(dt, tau):
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt debe ser finito y no negativo.")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("Las constantes de tiempo deben ser finitas y mayores que cero.")


def _spike_vector(spikes, size, name):
    values = np.asarray(spikes, dtype=float)
    if values.shape != (size,):
        raise ValueError(f"{name} debe tener forma ({size},), no {values.shape}.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} debe contener valores finitos no negativos.")
    return values


def _pair_plasticity(pre_spikes, post_spikes, trace_pre, trace_post, eligibility, decay):
    eligibility *= decay
    active_pre = np.flatnonzero(pre_spikes)
    active_post = np.flatnonzero(post_spikes)
    if active_pre.size and np.any(trace_post):
        eligibility[:, active_pre] -= trace_post[:, None] * pre_spikes[active_pre]
    if active_post.size and np.any(trace_pre):
        eligibility[active_post, :] += post_spikes[active_post, None] * trace_pre
    trace_pre *= decay
    trace_post *= decay
    trace_pre += pre_spikes
    trace_post += post_spikes


def _apply_weight_update(weights, eligibility, learning_rate, reward_signal, max_row_sum, mask=1.0):
    if reward_signal == 0.0:
        return
    weights += learning_rate * reward_signal * eligibility
    weights[:] = np.clip(weights, 0.0, 1.2) * mask
    row_sums = np.sum(weights, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_row_sum / np.maximum(row_sums, 1e-12))
    weights *= scale


class ChemicalSynapse:
    """Conductance filter with pair-based STDP and reward-modulated eligibility."""

    def __init__(
        self,
        pre_size,
        post_size,
        tau_syn=5.0,
        tau_stdp=20.0,
        weight_scale=5.0,
        lr=0.08,
        max_row_sum=3.2,
        polarity=1.0,
    ):
        _validate_sizes(pre_size, post_size)
        _validate_dynamics(0.0, tau_syn)
        _validate_dynamics(0.0, tau_stdp)
        if not np.isfinite(lr) or lr < 0.0:
            raise ValueError("lr debe ser finito y no negativo.")
        if not np.isfinite(weight_scale) or weight_scale < 0.0:
            raise ValueError("weight_scale debe ser finito y no negativo.")
        if not np.isfinite(max_row_sum) or max_row_sum <= 0.0:
            raise ValueError("max_row_sum debe ser finito y mayor que cero.")
        if polarity not in (-1.0, 1.0):
            raise ValueError("polarity debe ser +1.0 (excitadora) o -1.0 (inhibidora).")

        self.pre_size = pre_size
        self.post_size = post_size
        self.tau_syn = tau_syn
        self.tau_stdp = tau_stdp
        self.weight_scale = weight_scale
        self.lr = lr
        self.max_row_sum = max_row_sum
        self.polarity = polarity

        self.W = np.zeros((post_size, pre_size), dtype=float)
        self.g = np.zeros(pre_size, dtype=float)
        self.E = np.zeros((post_size, pre_size), dtype=float)
        self.trace_pre = np.zeros(pre_size, dtype=float)
        self.trace_post = np.zeros(post_size, dtype=float)

    def step(self, pre_spikes, post_spikes=None, neuromod=None, dt=1.0, learn=True):
        _validate_dynamics(dt, self.tau_syn)
        pre_spikes = _spike_vector(pre_spikes, self.pre_size, "pre_spikes")
        self.g *= np.exp(-dt / self.tau_syn)
        self.g += pre_spikes
        current = self.polarity * (self.W @ self.g) * self.weight_scale

        if learn and neuromod is not None and post_spikes is not None:
            _validate_dynamics(dt, self.tau_stdp)
            post_spikes = _spike_vector(post_spikes, self.post_size, "post_spikes")
            _pair_plasticity(
                pre_spikes,
                post_spikes,
                self.trace_pre,
                self.trace_post,
                self.E,
                np.exp(-dt / self.tau_stdp),
            )
            _apply_weight_update(
                self.W,
                self.E,
                self.lr,
                neuromod.reward_signal,
                self.max_row_sum,
            )

        return current


class StochasticReservoirSynapse:
    """Sparse conductance synapse with reward-gated STDP and structural plasticity."""

    def __init__(
        self,
        pre_size,
        post_size,
        density=0.4,
        weight_scale=35.0,
        tau_s=5.0,
        tau_syn=None,
        tau_e=25.0,
        lr=0.05,
        max_row_sum=2.2,
        polarity=1.0,
    ):
        _validate_sizes(pre_size, post_size)
        effective_tau = tau_syn if tau_syn is not None else tau_s
        _validate_dynamics(0.0, effective_tau)
        _validate_dynamics(0.0, tau_e)
        if not np.isfinite(density) or not 0.0 <= density <= 1.0:
            raise ValueError("density debe estar entre cero y uno.")
        if not np.isfinite(lr) or lr < 0.0:
            raise ValueError("lr debe ser finito y no negativo.")
        if not np.isfinite(weight_scale) or weight_scale < 0.0:
            raise ValueError("weight_scale debe ser finito y no negativo.")
        if not np.isfinite(max_row_sum) or max_row_sum <= 0.0:
            raise ValueError("max_row_sum debe ser finito y mayor que cero.")
        if polarity not in (-1.0, 1.0):
            raise ValueError("polarity debe ser +1.0 (excitadora) o -1.0 (inhibidora).")
        self.pre_size = pre_size
        self.post_size = post_size
        self.weight_scale = weight_scale
        self.tau_s = effective_tau
        self.tau_e = tau_e
        self.lr = lr
        self.max_row_sum = max_row_sum
        self.polarity = polarity

        self.connectivity_mask = (np.random.rand(post_size, pre_size) < density).astype(float)
        for row in range(post_size):
            if not np.any(self.connectivity_mask[row]):
                self.connectivity_mask[row, np.random.randint(0, pre_size)] = 1.0

        self.W = np.random.uniform(0.1, 0.35, (post_size, pre_size)) * self.connectivity_mask
        self.g = np.zeros(post_size, dtype=float)
        self.E = np.zeros((post_size, pre_size), dtype=float)
        self.trace_pre = np.zeros(pre_size, dtype=float)
        self.trace_post = np.zeros(post_size, dtype=float)

    def step(self, pre_spikes, post_spikes=None, neuromod=None, dt=1.0, learn=True):
        _validate_dynamics(dt, self.tau_s)
        pre_spikes = _spike_vector(pre_spikes, self.pre_size, "pre_spikes")

        self.g *= np.exp(-dt / self.tau_s)
        active_pre = np.flatnonzero(pre_spikes)
        if active_pre.size:
            self.g += self.W[:, active_pre] @ pre_spikes[active_pre]

        if learn and post_spikes is not None and neuromod is not None:
            _validate_dynamics(dt, self.tau_e)
            post_spikes = _spike_vector(post_spikes, self.post_size, "post_spikes")
            _pair_plasticity(
                pre_spikes,
                post_spikes,
                self.trace_pre,
                self.trace_post,
                self.E,
                np.exp(-dt / self.tau_e),
            )

            reward_signal = neuromod.reward_signal
            if (
                reward_signal > 0.25
                and neuromod.attention_level > 0.15
                and active_pre.size
                and np.any(post_spikes)
            ):
                active_post = np.flatnonzero(post_spikes)
                candidates = self.connectivity_mask[np.ix_(active_post, active_pre)] == 0.0
                growth_probability = min(
                    0.25, reward_signal * neuromod.attention_level * 0.5
                )
                new_connections = candidates & (np.random.rand(*candidates.shape) < growth_probability)
                rows, columns = np.where(new_connections)
                if rows.size:
                    post_indices = active_post[rows]
                    pre_indices = active_pre[columns]
                    self.connectivity_mask[post_indices, pre_indices] = 1.0
                    self.W[post_indices, pre_indices] = 0.28
                    self.E[post_indices, pre_indices] = 1.0

            plasticity_gate = self.lr * (
                0.2 + 0.8 * neuromod.attention_level
            )
            _apply_weight_update(
                self.W,
                self.E,
                plasticity_gate,
                reward_signal,
                self.max_row_sum,
                self.connectivity_mask,
            )

            dominant_columns = np.max(self.W, axis=0) > 0.65
            if np.any(dominant_columns):
                winners = np.argmax(self.W, axis=0)
                losers = np.arange(self.post_size)[:, None] != winners[None, :]
                self.W *= np.where(dominant_columns[None, :] & losers, 0.9, 1.0)

            pruned = (self.W < 0.02) & (self.connectivity_mask == 1.0)
            can_prune = np.sum(self.connectivity_mask, axis=1) > 2
            prune_mask = pruned & can_prune[:, None]
            self.connectivity_mask[prune_mask] = 0.0
            self.W[prune_mask] = 0.0

        return self.polarity * self.g * self.weight_scale


class SparseReservoirSynapse:
    """Reservoir con pesos y elegibilidad almacenados solo para conexiones activas."""

    def __init__(
        self,
        pre_size,
        post_size,
        density=0.25,
        weight_scale=35.0,
        tau_syn=5.0,
        tau_e=25.0,
        lr=0.05,
        max_row_sum=2.2,
        polarity=1.0,
        block_size=65536,
        num_dendrites=None,
    ):
        _validate_sizes(pre_size, post_size)
        _validate_dynamics(0.0, tau_syn)
        _validate_dynamics(0.0, tau_e)
        if not np.isfinite(density) or not 0.0 < density <= 1.0:
            raise ValueError("density debe ser mayor que cero y menor o igual a uno.")
        if not np.isfinite(lr) or lr < 0.0:
            raise ValueError("lr debe ser finito y no negativo.")
        if not np.isfinite(weight_scale) or weight_scale < 0.0:
            raise ValueError("weight_scale debe ser finito y no negativo.")
        if not np.isfinite(max_row_sum) or max_row_sum <= 0.0:
            raise ValueError("max_row_sum debe ser finito y mayor que cero.")
        if polarity not in (-1.0, 1.0):
            raise ValueError("polarity debe ser +1.0 (excitadora) o -1.0 (inhibidora).")
        if (
            not isinstance(block_size, (int, np.integer))
            or isinstance(block_size, bool)
            or block_size <= 0
        ):
            raise ValueError("block_size debe ser un entero mayor que cero.")
        if num_dendrites is not None and (
            not isinstance(num_dendrites, (int, np.integer))
            or isinstance(num_dendrites, bool)
            or num_dendrites <= 0
        ):
            raise ValueError("num_dendrites debe ser un entero mayor que cero.")

        self.pre_size = int(pre_size)
        self.post_size = int(post_size)
        self.num_dendrites = (
            None if num_dendrites is None else min(int(num_dendrites), self.pre_size)
        )
        self.weight_scale = float(weight_scale)
        self.tau_s = float(tau_syn)
        self.tau_e = float(tau_e)
        self.lr = float(lr)
        self.max_row_sum = float(max_row_sum)
        self.polarity = float(polarity)
        self.block_size = int(block_size)
        pre_groups = []
        post_groups = []
        for post_idx in range(self.post_size):
            connection_count = max(
                1,
                int(np.random.binomial(self.pre_size, density)),
            )
            selected_pre = np.random.choice(
                self.pre_size,
                size=min(
                    connection_count,
                    self.pre_size,
                    self.num_dendrites or self.pre_size,
                ),
                replace=False,
            )
            pre_groups.append(selected_pre.astype(np.int32))
            post_groups.append(
                np.full(selected_pre.size, post_idx, dtype=np.int32)
            )
        pre_indices = np.concatenate(pre_groups)
        post_indices = np.concatenate(post_groups)
        max_dendrites = self.num_dendrites or self.pre_size
        connected_inputs = np.zeros(self.pre_size, dtype=bool)
        connected_inputs[pre_indices] = True
        missing_inputs = np.flatnonzero(~connected_inputs)
        connected_per_post = np.bincount(
            post_indices, minlength=self.post_size
        )
        for pre_idx in missing_inputs:
            available_posts = np.flatnonzero(
                connected_per_post < max_dendrites
            )
            if available_posts.size:
                post_idx = int(np.random.choice(available_posts))
                pre_indices = np.append(pre_indices, pre_idx)
                post_indices = np.append(post_indices, post_idx)
                connected_per_post[post_idx] += 1
                continue

            if self.post_size * max_dendrites < self.pre_size:
                break
            input_counts = np.bincount(pre_indices, minlength=self.pre_size)
            replaceable_edges = np.flatnonzero(input_counts[pre_indices] > 1)
            if not replaceable_edges.size:
                break
            edge_idx = int(np.random.choice(replaceable_edges))
            input_counts[pre_indices[edge_idx]] -= 1
            pre_indices[edge_idx] = pre_idx
            input_counts[pre_idx] += 1
        order = np.argsort(pre_indices, kind="stable")
        self.pre_indices = pre_indices[order]
        self.post_indices = post_indices[order]
        self.weights = np.random.uniform(
            0.1, 0.35, self.pre_indices.size
        ).astype(np.float32)
        self.eligibility = np.zeros(self.pre_indices.size, dtype=np.float32)
        self._quantization_step = None
        self._g_scale = 1.0
        self._eligibility_scale = 1.0
        self._trace_pre_scale = 1.0
        self._trace_post_scale = 1.0
        self.g = np.zeros(self.post_size, dtype=np.float32)
        self.trace_pre = np.zeros(self.pre_size, dtype=np.float32)
        self.trace_post = np.zeros(self.post_size, dtype=np.float32)
        self._rebuild_indices()

    def _rebuild_indices(self):
        self.pre_offsets = np.searchsorted(
            self.pre_indices, np.arange(self.pre_size + 1), side="left"
        )
        post_order = np.argsort(self.post_indices, kind="stable")
        self.post_edge_indices = post_order.astype(np.int32)
        sorted_posts = self.post_indices[post_order]
        self.post_offsets = np.searchsorted(
            sorted_posts, np.arange(self.post_size + 1), side="left"
        )

    def set_num_dendrites(self, num_dendrites):
        """Limita el número de entradas presinápticas de cada neurona destino."""
        if (
            not isinstance(num_dendrites, (int, np.integer))
            or isinstance(num_dendrites, bool)
            or num_dendrites <= 0
        ):
            raise ValueError("num_dendrites debe ser un entero mayor que cero.")
        self.dequantize_weights()
        self.num_dendrites = min(int(num_dendrites), self.pre_size)
        keep = np.zeros(self.connection_count, dtype=bool)
        for post_idx in range(self.post_size):
            start, end = self.post_offsets[post_idx : post_idx + 2]
            edges = self.post_edge_indices[start:end]
            if edges.size > self.num_dendrites:
                selected = np.random.choice(
                    edges, size=self.num_dendrites, replace=False
                )
                keep[selected] = True
            else:
                keep[edges] = True
        if not np.all(keep):
            self.pre_indices = self.pre_indices[keep]
            self.post_indices = self.post_indices[keep]
            self.weights = self.weights[keep]
            self.eligibility = self.eligibility[keep]
            self._rebuild_indices()

    @staticmethod
    def _edge_indices(offsets, edge_order, neurons):
        if edge_order is None:
            chunks = [
                np.arange(offsets[index], offsets[index + 1], dtype=np.int32)
                for index in neurons
                if offsets[index] < offsets[index + 1]
            ]
        else:
            chunks = [
                edge_order[offsets[index] : offsets[index + 1]]
                for index in neurons
                if offsets[index] < offsets[index + 1]
            ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int32)

    def _weights_at(self, edge_indices):
        if self._quantization_step is None:
            return self.weights[edge_indices]
        return self.weights[edge_indices].astype(np.float32) * self._quantization_step

    def quantize_weights(self):
        """Guarda pesos con cuatro decimales en entero fijo de 16 bits."""
        if self._quantization_step is None:
            self._quantization_step = 1e-4
            self.weights = np.rint(
                np.clip(self.weights, 0.0, 1.2) * 10_000
            ).astype(np.uint16)

    def dequantize_weights(self):
        """Restaura pesos float32 para continuar entrenando."""
        if self._quantization_step is not None:
            self.weights = self._weights_at(slice(None)).copy()
            self._quantization_step = None

    @property
    def has_pending_current(self):
        return bool(self._g_scale > 1e-7 and np.any(self.g))

    def _stabilize_lazy_scales(self):
        for name, values in (
            ("_g_scale", self.g),
            ("_eligibility_scale", self.eligibility),
            ("_trace_pre_scale", self.trace_pre),
            ("_trace_post_scale", self.trace_post),
        ):
            scale = getattr(self, name)
            if scale < 1e-12:
                values *= scale
                setattr(self, name, 1.0)

    def _has_connection(self, pre_idx, post_idx):
        start, end = self.pre_offsets[pre_idx : pre_idx + 2]
        return bool(np.any(self.post_indices[start:end] == post_idx))

    @property
    def connection_count(self):
        return int(self.weights.size)

    @property
    def memory_usage_bytes(self):
        arrays = (
            self.pre_indices,
            self.post_indices,
            self.weights,
            self.eligibility,
            self.g,
            self.trace_pre,
            self.trace_post,
            self.pre_offsets,
            self.post_edge_indices,
            self.post_offsets,
        )
        return sum(array.nbytes for array in arrays)

    def dense_weights(self):
        matrix = np.zeros((self.post_size, self.pre_size), dtype=np.float32)
        matrix[self.post_indices, self.pre_indices] = self._weights_at(slice(None))
        return matrix

    def state_dict(self):
        return {
            "pre_indices": self.pre_indices.copy(),
            "post_indices": self.post_indices.copy(),
            "weights": self._weights_at(slice(None)).copy(),
        }

    def load_state(self, pre_indices, post_indices, weights):
        pre_indices = np.asarray(pre_indices, dtype=np.int32)
        post_indices = np.asarray(post_indices, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float32)
        if (
            pre_indices.ndim != 1
            or post_indices.shape != pre_indices.shape
            or weights.shape != pre_indices.shape
            or not np.all(np.isfinite(weights))
            or np.any(pre_indices < 0)
            or np.any(pre_indices >= self.pre_size)
            or np.any(post_indices < 0)
            or np.any(post_indices >= self.post_size)
        ):
            raise ValueError("El estado disperso de pesos contiene forma o valores inválidos.")
        order = np.argsort(pre_indices, kind="stable")
        self.pre_indices = pre_indices[order].copy()
        self.post_indices = post_indices[order].copy()
        self.weights = weights[order].copy()
        self._quantization_step = None
        self.eligibility = np.zeros(self.weights.size, dtype=np.float32)
        self._g_scale = 1.0
        self._eligibility_scale = 1.0
        self._trace_pre_scale = 1.0
        self._trace_post_scale = 1.0
        self.g.fill(0.0)
        self.trace_pre.fill(0.0)
        self.trace_post.fill(0.0)
        self._rebuild_indices()
        if self.num_dendrites is not None:
            self.set_num_dendrites(self.num_dendrites)

    def apply_supervised_update(self, error, pre_rate, learning_rate):
        self.dequantize_weights()
        error = np.asarray(error, dtype=float)
        if error.shape != (self.post_size,) or not np.all(np.isfinite(error)):
            raise ValueError(f"error debe tener forma ({self.post_size},) y ser finito.")
        pre_rate = _spike_vector(pre_rate, self.pre_size, "pre_rate")
        active_pre = np.flatnonzero(pre_rate)
        growing_post = np.flatnonzero(error > 0.0)
        if active_pre.size and growing_post.size:
            new_pairs = [
                (int(pre_idx), int(post_idx))
                for post_idx in growing_post
                for pre_idx in active_pre
                if not self._has_connection(int(pre_idx), int(post_idx))
            ]
            if new_pairs:
                new_pre, new_post = zip(*new_pairs)
                self._append_connections(new_pre, new_post, initial_weight=0.08)
        edge_indices = self._edge_indices(
            self.pre_offsets,
            None,
            active_pre,
        )
        if edge_indices.size == 0:
            return
        edge_posts = self.post_indices[edge_indices]
        edge_delta = error[edge_posts] * pre_rate[self.pre_indices[edge_indices]]
        self.weights[edge_indices] += (learning_rate * edge_delta).astype(np.float32)
        self.weights[edge_indices] = np.clip(self.weights[edge_indices], 0.0, 1.2)
        affected_posts = np.unique(edge_posts)
        affected_edges = self._edge_indices(
            self.post_offsets,
            self.post_edge_indices,
            affected_posts,
        )
        affected_edge_posts = self.post_indices[affected_edges]
        local_rows = np.searchsorted(affected_posts, affected_edge_posts)
        row_sums = np.bincount(
            local_rows,
            weights=self.weights[affected_edges],
            minlength=affected_posts.size,
        )
        scale = np.minimum(
            1.0,
            self.max_row_sum / np.maximum(row_sums, 1e-12),
        )
        self.weights[affected_edges] *= scale[local_rows].astype(np.float32)

    def _append_connections(self, pre_indices, post_indices, initial_weight=0.28):
        if not len(pre_indices):
            return
        self.dequantize_weights()
        self._stabilize_lazy_scales()
        pre_indices = np.asarray(pre_indices, dtype=np.int32)
        post_indices = np.asarray(post_indices, dtype=np.int32)
        if self.num_dendrites is not None:
            connected_per_post = np.bincount(
                self.post_indices, minlength=self.post_size
            )
            accepted_pre = []
            accepted_post = []
            candidate_pairs = set()
            for pre_idx, post_idx in zip(pre_indices, post_indices):
                pair = (int(pre_idx), int(post_idx))
                if pair in candidate_pairs or self._has_connection(*pair):
                    continue
                candidate_pairs.add(pair)
                if connected_per_post[post_idx] < self.num_dendrites:
                    accepted_pre.append(pre_idx)
                    accepted_post.append(post_idx)
                    connected_per_post[post_idx] += 1
            pre_indices = np.asarray(accepted_pre, dtype=np.int32)
            post_indices = np.asarray(accepted_post, dtype=np.int32)
            if pre_indices.size == 0:
                return
        self.pre_indices = np.concatenate(
            (self.pre_indices, pre_indices)
        )
        self.post_indices = np.concatenate(
            (self.post_indices, post_indices)
        )
        self.weights = np.concatenate(
            (
                self.weights,
                np.full(pre_indices.size, initial_weight, dtype=np.float32),
            )
        )
        self.eligibility = np.concatenate(
            (
                self.eligibility,
                np.full(
                    pre_indices.size,
                    1.0 / self._eligibility_scale,
                    dtype=np.float32,
                ),
            )
        )
        order = np.argsort(self.pre_indices, kind="stable")
        self.pre_indices = self.pre_indices[order]
        self.post_indices = self.post_indices[order]
        self.weights = self.weights[order]
        self.eligibility = self.eligibility[order]
        self._rebuild_indices()

    def step(self, pre_spikes, post_spikes=None, neuromod=None, dt=1.0, learn=True):
        _validate_dynamics(dt, self.tau_s)
        pre_spikes = _spike_vector(pre_spikes, self.pre_size, "pre_spikes")
        decay_g = np.exp(-dt / self.tau_s)
        self._g_scale *= decay_g
        self._stabilize_lazy_scales()
        active_pre = np.flatnonzero(pre_spikes)
        pre_edges = self._edge_indices(
            self.pre_offsets,
            None,
            active_pre,
        )
        for pre_idx in active_pre:
            start, end = self.pre_offsets[pre_idx : pre_idx + 2]
            for block_start in range(start, end, self.block_size):
                block_end = min(block_start + self.block_size, end)
                np.add.at(
                    self.g,
                    self.post_indices[block_start:block_end],
                    self._weights_at(slice(block_start, block_end))
                    * pre_spikes[pre_idx]
                    / self._g_scale,
                )

        if learn and post_spikes is not None and neuromod is not None:
            self.dequantize_weights()
            _validate_dynamics(dt, self.tau_e)
            post_spikes = _spike_vector(post_spikes, self.post_size, "post_spikes")
            decay = np.exp(-dt / self.tau_e)
            self._eligibility_scale *= decay
            self._trace_pre_scale *= decay
            self._trace_post_scale *= decay
            self._stabilize_lazy_scales()
            active_post = np.flatnonzero(post_spikes)
            post_edges = self._edge_indices(
                self.post_offsets,
                self.post_edge_indices,
                active_post,
            )
            if pre_edges.size:
                edge_posts = self.post_indices[pre_edges]
                self.eligibility[pre_edges] -= (
                    self.trace_post[edge_posts]
                    * self._trace_post_scale
                    * pre_spikes[self.pre_indices[pre_edges]]
                    / self._eligibility_scale
                )
            if post_edges.size:
                edge_pres = self.pre_indices[post_edges]
                self.eligibility[post_edges] += (
                    post_spikes[self.post_indices[post_edges]]
                    * self.trace_pre[edge_pres]
                    * self._trace_pre_scale
                    / self._eligibility_scale
                )
            if active_pre.size:
                self.trace_pre[active_pre] += (
                    pre_spikes[active_pre] / self._trace_pre_scale
                )
            if active_post.size:
                self.trace_post[active_post] += (
                    post_spikes[active_post] / self._trace_post_scale
                )

            reward = neuromod.reward_signal
            if (
                reward > 0.25
                and neuromod.attention_level > 0.15
                and active_pre.size
                and np.any(post_spikes)
            ):
                growth_probability = min(
                    0.25, reward * neuromod.attention_level * 0.5
                )
                new_pairs = []
                connected_per_post = np.bincount(
                    self.post_indices, minlength=self.post_size
                )
                for post_idx in active_post:
                    available = (
                        active_pre.size
                        if self.num_dendrites is None
                        else max(
                            0,
                            self.num_dendrites - connected_per_post[post_idx],
                        )
                    )
                    candidates = [
                        int(pre_idx)
                        for pre_idx in active_pre
                        if not self._has_connection(int(pre_idx), int(post_idx))
                        and np.random.random() < growth_probability
                    ]
                    if available and candidates:
                        selected = np.random.choice(
                            candidates,
                            size=min(available, len(candidates)),
                            replace=False,
                        )
                        new_pairs.extend(
                            (int(pre_idx), int(post_idx))
                            for pre_idx in selected
                        )
                if new_pairs:
                    new_pre, new_post = zip(*new_pairs)
                    self._append_connections(new_pre, new_post)

            if reward != 0.0:
                plasticity_gate = self.lr * (
                    0.2 + 0.8 * neuromod.attention_level
                )
                self.weights += (
                    plasticity_gate
                    * reward
                    * self.eligibility
                    * self._eligibility_scale
                ).astype(np.float32)
                self.weights[:] = np.clip(self.weights, 0.0, 1.2)
                row_sums = np.bincount(
                    self.post_indices,
                    weights=self.weights,
                    minlength=self.post_size,
                )
                self.weights *= np.minimum(
                    1.0,
                    self.max_row_sum / np.maximum(row_sums, 1e-12),
                )[self.post_indices].astype(np.float32)

                dominant = self.weights > 0.65
                if np.any(dominant):
                    column_max = np.full(self.pre_size, -np.inf)
                    np.maximum.at(column_max, self.pre_indices, self.weights)
                    winners = np.full(self.pre_size, -1, dtype=np.int32)
                    for edge_idx in np.flatnonzero(
                        self.weights == column_max[self.pre_indices]
                    ):
                        winners[self.pre_indices[edge_idx]] = self.post_indices[edge_idx]
                    losers = dominant & (
                        self.post_indices != winners[self.pre_indices]
                    )
                    self.weights[losers] *= 0.9

                row_counts = np.bincount(
                    self.post_indices, minlength=self.post_size
                )
                prune = (self.weights < 0.02) & (
                    row_counts[self.post_indices] > 2
                )
                if np.any(prune):
                    keep = ~prune
                    self.pre_indices = self.pre_indices[keep]
                    self.post_indices = self.post_indices[keep]
                    self.weights = self.weights[keep]
                    self.eligibility = self.eligibility[keep]
                    self._rebuild_indices()

        return self.polarity * self.g * self._g_scale * self.weight_scale


class RecurrentReservoirSynapse(StochasticReservoirSynapse):
    def __init__(
        self,
        size,
        density=0.3,
        tau_syn=15.0,
        weight_scale=4.0,
        lr=0.02,
        polarity=1.0,
    ):
        super().__init__(
            pre_size=size,
            post_size=size,
            density=density,
            tau_s=tau_syn,
            weight_scale=weight_scale,
            lr=lr,
            max_row_sum=1.2,
            polarity=polarity,
        )
        np.fill_diagonal(self.W, 0.0)
        self.connectivity_mask[np.arange(size), np.arange(size)] = 0.0
        self.W *= self.connectivity_mask
