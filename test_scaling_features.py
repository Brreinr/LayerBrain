import unittest
from types import SimpleNamespace

import numpy as np

from layerbrain import (
    BrainNetwork,
    BrainTrainer,
    CochlearAuditoryLayer,
    DentateGranuleLayer,
    GustatoryLayer,
    LIFLayer,
    OlfactoryLayer,
    PyramidalLayer,
    RetinalVisualLayer,
    SomatosensoryLayer,
    SparseReservoirSynapse,
    StriatalMSNLayer,
    TextSensoryLayer,
    ThalamicLayer,
    TrainingError,
)


class ScalingFeatureTests(unittest.TestCase):
    @staticmethod
    def _make_trainer():
        network = BrainNetwork()
        layers = [network.add_layer(LIFLayer(3, noise_std=0.0)) for _ in range(3)]
        for source, target in ((0, 1), (1, 2), (0, 2)):
            network.connect(
                source,
                target,
                SparseReservoirSynapse(3, 3, density=1.0),
            )
        return BrainTrainer(network), network

    @staticmethod
    def _one_training_sample():
        return [
            {
                "inputs": {0: np.array([1.0, 0.0, 0.0])},
                "target_layer": 2,
                "target_neuron": 0,
            }
        ]

    def test_lineal_training_keeps_only_adjacent_layer_connections(self):
        trainer, network = self._make_trainer()

        history = trainer.fit_supervised(
            self._one_training_sample(),
            epochs=1,
            steps_per_sample=1,
            lineal=True,
        )

        self.assertTrue(history["lineal"])
        self.assertEqual(
            [(source, target) for source, target, _ in network.connections],
            [(0, 1), (1, 2)],
        )
        route = network.select_route(2, source_layers={0})
        self.assertEqual(
            {
                (network.connections[index][0], network.connections[index][1])
                for index in route
            },
            {(0, 1), (1, 2)},
        )

    def test_non_lineal_training_builds_recurrent_all_to_all_topology(self):
        trainer, network = self._make_trainer()

        history = trainer.fit_supervised(
            self._one_training_sample(),
            epochs=1,
            steps_per_sample=1,
            lineal=False,
        )

        self.assertFalse(history["lineal"])
        self.assertEqual(len(network.connections), len(network.layers) ** 2)
        self.assertIn((1, 1), [(source, target) for source, target, _ in network.connections])
        route = network.select_route(2, source_layers={0})
        self.assertTrue(
            any(network.connections[index][0] == 0 for index in route)
        )

    def test_lineal_prediction_decodes_subthreshold_output_without_spike_error(self):
        network = BrainNetwork()
        output_layer = network.add_layer(
            LIFLayer(
                2,
                v_rest=0.0,
                v_thresh=1.0,
                noise_std=0.0,
            )
        )
        network.lineal_mode = True

        winner, spike_counts = network.predict(
            {output_layer: np.array([0.1, 0.5])},
            output_layer_idx=output_layer,
            steps=1,
        )

        self.assertEqual(winner, 1)
        np.testing.assert_array_equal(spike_counts, np.zeros(2, dtype=int))

    def test_lineal_training_rejects_non_boolean_values(self):
        trainer, _ = self._make_trainer()

        with self.assertRaises(TrainingError):
            trainer.fit_supervised(
                self._one_training_sample(),
                epochs=1,
                steps_per_sample=1,
                lineal=1,
            )

    def test_lineal_training_does_not_early_stop_on_unchanging_route(self):
        trainer, _ = self._make_trainer()
        callback_calls = []

        def stop_on_first_epoch(summary):
            callback_calls.append(summary)
            return True

        history = trainer.fit_supervised(
            self._one_training_sample(),
            epochs=3,
            steps_per_sample=1,
            route_callback=stop_on_first_epoch,
            lineal=True,
        )

        self.assertEqual(len(callback_calls), 3)
        self.assertEqual(len(history["accuracy"]), 3)
        self.assertFalse(history["stopped_early"])

    def test_all_neuron_layer_types_accept_custom_dendrite_count(self):
        layers = (
            LIFLayer(2, num_dendrites=3),
            TextSensoryLayer(2, vocabulary=("a", "b"), num_dendrites=3),
            ThalamicLayer(2, num_dendrites=3),
            DentateGranuleLayer(2, num_dendrites=3),
            PyramidalLayer(2, num_dendrites=3),
            StriatalMSNLayer(2, num_dendrites=3),
            RetinalVisualLayer(2, 2, num_dendrites=3),
            CochlearAuditoryLayer(2, num_dendrites=3),
            SomatosensoryLayer(2, num_dendrites=3),
            OlfactoryLayer(2, num_dendrites=3),
            GustatoryLayer(2, num_dendrites=3),
        )

        self.assertTrue(all(layer.num_dendrites == 3 for layer in layers))

    def test_layer_dendrites_limit_connected_presynaptic_neurons(self):
        target = LIFLayer(4, num_dendrites=2)
        self.assertEqual(target.num_dendrites, 2)
        with self.assertRaises(ValueError):
            LIFLayer(4, num_dendrites=0)

        network = BrainNetwork()
        source = network.add_layer(LIFLayer(12))
        destination = network.add_layer(target)
        synapse = SparseReservoirSynapse(12, 4, density=1.0)
        network.connect(source, destination, synapse)

        connection_counts = np.bincount(
            synapse.post_indices,
            minlength=target.num_neurons,
        )
        self.assertTrue(np.all(connection_counts <= target.num_dendrites))

    def test_dendrite_limit_is_shared_across_multiple_incoming_layers(self):
        network = BrainNetwork()
        sources = [network.add_layer(LIFLayer(8)) for _ in range(3)]
        destination = network.add_layer(LIFLayer(4, num_dendrites=3))
        for source in sources:
            network.connect(
                source,
                destination,
                SparseReservoirSynapse(8, 4, density=1.0),
            )

        total_incoming = np.zeros(4, dtype=int)
        for _, target, synapse in network.connections:
            if target == destination:
                total_incoming += np.bincount(
                    synapse.post_indices,
                    minlength=4,
                )
        self.assertTrue(np.all(total_incoming <= 3))

    def test_learning_cannot_grow_past_dendrite_limit(self):
        synapse = SparseReservoirSynapse(
            12,
            4,
            density=0.01,
            num_dendrites=2,
        )
        synapse.apply_supervised_update(
            np.ones(4),
            np.ones(12),
            learning_rate=0.1,
        )

        connection_counts = np.bincount(
            synapse.post_indices,
            minlength=synapse.post_size,
        )
        self.assertTrue(np.all(connection_counts <= 2))

    def test_sparse_connectivity_covers_presynaptic_neurons_when_capacity_allows(self):
        synapse = SparseReservoirSynapse(
            8,
            4,
            density=0.01,
            num_dendrites=2,
        )

        connected_per_input = np.bincount(
            synapse.pre_indices,
            minlength=synapse.pre_size,
        )
        connected_per_neuron = np.bincount(
            synapse.post_indices,
            minlength=synapse.post_size,
        )
        self.assertTrue(np.all(connected_per_input > 0))
        self.assertTrue(np.all(connected_per_neuron <= 2))

    def test_sparse_synapse_processes_connections_in_configured_blocks(self):
        synapse = SparseReservoirSynapse(
            4,
            3,
            density=1.0,
            block_size=2,
        )

        current = synapse.step(np.array([1.0, 0.0, 0.0, 0.0]), learn=False)

        self.assertEqual(synapse.block_size, 2)
        self.assertEqual(current.shape, (3,))
        self.assertTrue(np.all(np.isfinite(current)))
        with self.assertRaises(ValueError):
            SparseReservoirSynapse(4, 3, block_size=0)

    def test_sparse_synapse_decay_is_lazy_but_numerically_preserved(self):
        synapse = SparseReservoirSynapse(
            4,
            3,
            density=1.0,
            tau_syn=4.0,
            tau_e=5.0,
        )
        neuromod = SimpleNamespace(reward_signal=0.0, attention_level=0.0)
        pre = np.array([1.0, 0.0, 0.0, 0.0])
        post = np.zeros(3)

        synapse.step(pre, post, neuromod=neuromod, dt=1.0, learn=True)
        initial_conductance = synapse.g * synapse._g_scale
        initial_trace = synapse.trace_pre[0] * synapse._trace_pre_scale
        initial_eligibility = synapse.eligibility * synapse._eligibility_scale

        synapse.step(np.zeros(4), post, neuromod=neuromod, dt=1.0, learn=True)
        np.testing.assert_allclose(
            synapse.g * synapse._g_scale,
            initial_conductance * np.exp(-1.0 / synapse.tau_s),
        )
        self.assertAlmostEqual(
            synapse.trace_pre[0] * synapse._trace_pre_scale,
            initial_trace * np.exp(-1.0 / synapse.tau_e),
        )
        np.testing.assert_allclose(
            synapse.eligibility * synapse._eligibility_scale,
            initial_eligibility * np.exp(-1.0 / synapse.tau_e),
        )

    def test_quantized_inference_reduces_weight_storage(self):
        synapse = SparseReservoirSynapse(16, 8, density=0.5)
        original = synapse.dense_weights()
        float_bytes = synapse.weights.nbytes

        synapse.quantize_weights()
        quantized = synapse.dense_weights()

        self.assertEqual(synapse.weights.dtype, np.uint16)
        self.assertEqual(synapse.weights.nbytes * 2, float_bytes)
        self.assertLessEqual(
            float(np.max(np.abs(original - quantized))),
            0.00005 + 1e-7,
        )
        np.testing.assert_allclose(quantized, np.round(original, 4), atol=1e-7)
        self.assertTrue(np.all(synapse.weights <= 12_000))
        synapse.dequantize_weights()
        self.assertEqual(synapse.weights.dtype, np.float32)

    def test_supervised_sparse_update_only_changes_active_input_edges(self):
        synapse = SparseReservoirSynapse(
            8,
            5,
            density=1.0,
            max_row_sum=100.0,
        )
        before = synapse.weights.copy()
        pre_rate = np.zeros(8)
        pre_rate[2] = 1.0
        error = np.ones(5)

        synapse.apply_supervised_update(error, pre_rate, learning_rate=0.1)

        changed = np.flatnonzero(synapse.weights != before)
        changed_pre = set(synapse.pre_indices[changed].tolist())
        self.assertTrue(changed_pre.issubset({2}))

    def test_lif_event_path_leaves_quiescent_neurons_at_rest(self):
        layer = LIFLayer(
            5,
            v_rest=0.0,
            v_thresh=1.0,
            v_reset=0.0,
            noise_std=0.0,
            t_ref=1.0,
        )
        current = np.zeros(5)
        current[3] = 2.0

        spikes = layer.step(current)

        self.assertEqual(spikes[0], 0)
        self.assertEqual(spikes[1], 0)
        self.assertEqual(spikes[2], 0)
        self.assertEqual(spikes[4], 0)
        self.assertEqual(layer.v[0], 0.0)
        self.assertEqual(layer.v[1], 0.0)
        self.assertEqual(layer.v[2], 0.0)
        self.assertEqual(layer.v[4], 0.0)


if __name__ == "__main__":
    unittest.main()
