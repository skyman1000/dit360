"""Synthetic CPU checks; no user images, checkpoints or model downloads."""
import importlib.util
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from evaluation.metrics.places_is import probability_diagnostics
from evaluation.metrics.faed import resize_image


class DiagnosticTests(unittest.TestCase):
    def test_is_entropy_identity_and_order_effect(self):
        # Five scenes, each internally identical. Ordered splits hide diversity.
        p = np.repeat(np.eye(5), 10, axis=0)
        rows = [{'id': 'a' * 11 + '_' + f'{i:032x}'} for i in range(len(p))]
        result = probability_diagnostics(p, rows)
        self.assertAlmostEqual(result['global_is'], 5)
        self.assertAlmostEqual(result['exp_entropy_difference'], 5)
        self.assertAlmostEqual(result['split_sensitivity']['manifest_order/splits5']['value'], 1)
        self.assertGreater(result['split_sensitivity']['shuffled_seed0/splits5']['value'], 2)
        self.assertEqual(result, probability_diagnostics(p, rows))
        self.assertEqual(result['per_building']['a' * 11]['n'], 50)

    def test_is_no_fake_diversity_or_invalid_probabilities(self):
        p = np.full((3, 5), .2)
        result = probability_diagnostics(p, [{}] * 3)
        self.assertAlmostEqual(result['global_is'], 1)
        self.assertEqual(result['per_building'], {})
        self.assertEqual(len(result['split_sensitivity']), 2)
        with self.assertRaises(ValueError):
            probability_diagnostics(p * 2, [{}] * 3)

    def test_legacy_faed_pixels_unchanged(self):
        image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (70, 140, 3), dtype=np.uint8))
        expected = np.array(image.resize((64, 32), Image.Resampling.BICUBIC))
        for label in ('real', 'generated'):
            np.testing.assert_array_equal(resize_image(image, 32, label, 'pil-bicubic'), expected)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV optional')
    def test_panfusion_resize_matches_upstream_and_keeps_rgb(self):
        import cv2
        pixels = np.random.default_rng(8).integers(0, 256, (70, 140, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        np.testing.assert_array_equal(resize_image(image, 32, 'real', 'panfusion'),
                                      cv2.resize(pixels, (64, 32), interpolation=cv2.INTER_AREA))
        np.testing.assert_array_equal(resize_image(image, 32, 'generated', 'panfusion'),
                                      cv2.resize(pixels, (64, 32)))

    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('py360convert'),
                         'Projection test requires torch and py360convert')
    def test_actual_projection_pole_and_equator_centers(self):
        from evaluation.metrics.fid import views
        pixels = np.zeros((128, 256, 3), dtype=np.uint8)
        pixels[:32, :, 0] = 255
        pixels[32:96, :, 1] = 255
        pixels[96:, :, 2] = 255
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sphere.png'
            Image.fromarray(pixels).save(path)
            args = SimpleNamespace(face_size=32)
            poles = views(path, 'fidpole', args)
            np.testing.assert_array_equal(poles[0][:, 16, 16], [255, 0, 0])
            np.testing.assert_array_equal(poles[1][:, 16, 16], [0, 0, 255])
            for face in views(path, 'fidequ', args):
                np.testing.assert_array_equal(face[:, 16, 16], [0, 255, 0])

    def test_comparison_preserves_baseline_and_rejects_different_inputs(self):
        from evaluation.compare_reports import PAPER, comparison_rows
        baseline = {'complete': True, 'inputs_manifest_sha256': 'same',
                    'results': {k: {'status': 'ok', 'value': v + 1} for k, v in PAPER.items()}}
        regions = copy.deepcopy(baseline)
        regions['results']['faed']['value'] = 0  # must never overwrite baseline
        rows = comparison_rows(baseline, regions)
        self.assertEqual(len(rows), 11)
        self.assertEqual(next(r['current'] for r in rows if r['metric'] == 'faed'), 3.91)
        regions['inputs_manifest_sha256'] = 'different'
        with self.assertRaises(ValueError):
            comparison_rows(baseline, regions)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Fake network test requires torch')
    def test_is_artifact_roundtrip_keeps_generated_score(self):
        import torch
        from evaluation.metrics.places_is import compute
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = torch.nn.Linear(3, 365)
            weights = root / 'tiny.pt'
            torch.save(model.state_dict(), weights)
            args = SimpleNamespace(places_arch='resnet18', places_weights=weights,
                                   device='cpu', batch_size=1, is_splits=1,
                                   is_diagnostics=True, output=root)
            generated = np.eye(2)
            real = np.ones((2, 2)) / 2
            rows = [{'image': 'a.png', 'id': 'a'}, {'image': 'b.png', 'id': 'b'}]
            with patch('torchvision.models.resnet18', return_value=model), \
                    patch('evaluation.metrics.places_is.collect_probabilities', side_effect=[generated, real]):
                result = compute('is', rows, ['c.png', 'd.png'], args)
            self.assertEqual(result['value'], 2)
            diagnostics = json.loads((root / 'is_diagnostics.json').read_text())
            self.assertEqual(diagnostics['reference']['global_is'], 1)
            with np.load(root / 'is_probabilities_generated.npz', allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['probabilities'], generated)
                self.assertEqual(archive['ids'].tolist(), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
