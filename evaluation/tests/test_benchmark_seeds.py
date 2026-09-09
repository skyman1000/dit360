import unittest
from inference_benchmark import sample_seed


class SeedTests(unittest.TestCase):
    def test_shared_mode_preserves_original_seeds(self):
        self.assertEqual(sample_seed(0, 'a', 'shared'), 0)
        self.assertEqual(sample_seed(42, 'b', 'shared'), 42)

    def test_viewpoint_changes_seed_and_reordering_does_not(self):
        ids = ['scene_view1', 'scene_view2', 'scene_view3']
        seeds = {i: sample_seed(0, i, 'per-id') for i in ids}
        reordered = {i: sample_seed(0, i, 'per-id') for i in reversed(ids)}
        self.assertEqual(seeds, reordered)
        self.assertEqual(len(set(seeds.values())), 3)
        self.assertTrue(all(0 <= s < 2 ** 63 for s in seeds.values()))
        self.assertNotEqual(sample_seed(1, ids[0], 'per-id'), seeds[ids[0]])
