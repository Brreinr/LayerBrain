import tempfile
import unittest
from pathlib import Path

from layerbrain import PersistenceError, SpikingClassifier, TextDataset


class ClassifierPersistenceTests(unittest.TestCase):
    @staticmethod
    def _make_model(classes=("x", "y")):
        dataset = TextDataset(
            samples=(("a", "x"), ("b", "y")),
            prediction_samples=("a", "b"),
            classes=classes,
        )
        return SpikingClassifier(
            dataset=dataset,
            hidden_sizes=(4,),
            steps_per_sample=4,
            adaptive_routing=True,
            synapse_density=0.5,
        )

    def test_save_and_load_restores_weights_topology_and_route_scores(self):
        model = self._make_model()
        model.trainer.fit_dataset(
            model.sensor_dataset,
            epochs=1,
            steps_per_sample=model.steps_per_sample,
            lineal=False,
        )
        expected_pairs = [
            (pre_idx, post_idx)
            for pre_idx, post_idx, _ in model.network.connections
        ]
        expected_states = [
            synapse.state_dict()
            for _, _, synapse in model.network.connections
        ]
        model.network.route_scores = [
            float(index) / 10 for index in range(len(model.network.connections))
        ]
        expected_scores = model.network.route_scores.copy()

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "trained-model.npz"
            self.assertEqual(model.save(filepath), str(filepath))

            restored = self._make_model()
            self.assertIs(restored.load(filepath), restored)

        self.assertEqual(restored.network.lineal_mode, False)
        self.assertEqual(
            [
                (pre_idx, post_idx)
                for pre_idx, post_idx, _ in restored.network.connections
            ],
            expected_pairs,
        )
        self.assertEqual(restored.network.route_scores, expected_scores)
        for (_, _, synapse), expected in zip(
            restored.network.connections, expected_states
        ):
            actual = synapse.state_dict()
            for name in expected:
                self.assertTrue((actual[name] == expected[name]).all())
            self.assertEqual(actual["weights"].dtype, expected["weights"].dtype)

    def test_load_restores_linear_topology_in_new_classifier(self):
        model = self._make_model()
        model.trainer.fit_dataset(
            model.sensor_dataset,
            epochs=1,
            steps_per_sample=model.steps_per_sample,
            lineal=True,
        )
        expected_pairs = [
            (pre_idx, post_idx)
            for pre_idx, post_idx, _ in model.network.connections
        ]

        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "linear-model.npz"
            model.save(filepath)
            restored = self._make_model()
            restored.load(filepath)

        self.assertIs(restored.network.lineal_mode, True)
        self.assertEqual(
            [
                (pre_idx, post_idx)
                for pre_idx, post_idx, _ in restored.network.connections
            ],
            expected_pairs,
        )

    def test_load_rejects_model_with_different_class_order(self):
        model = self._make_model()
        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "model.npz"
            model.save(filepath)
            incompatible = self._make_model(classes=("y", "x"))
            with self.assertRaises(PersistenceError):
                incompatible.load(filepath)

    def test_save_uses_exact_path_without_appending_npz(self):
        model = self._make_model()
        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "model.data"
            model.save(filepath)
            self.assertTrue(filepath.is_file())
            self.assertFalse(Path(f"{filepath}.npz").exists())
