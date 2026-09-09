"""Test adapter data flow with tiny fake networks, never pretrained weights."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional CPU adapter tests require torch')
class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import before patch.dict(sys.modules), whose restoration removes new modules.
        import torch

    def test_cubemap_selects_poles_and_equator_as_distinct_groups(self):
        from evaluation.metrics.fid import views
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pano.png'
            Image.new('RGB', (64, 32)).save(path)
            values = dict(F=10, R=20, B=30, L=40, U=50, D=60)
            def e2c(image, face_w, mode, cube_format):
                self.assertEqual(cube_format, 'dict')
                self.assertEqual(mode, 'bilinear')
                return {k: np.full((face_w, face_w, 3), v, dtype=np.uint8) for k, v in values.items()}
            with patch.dict('sys.modules', {'py360convert': SimpleNamespace(e2c=e2c)}):
                args = SimpleNamespace(face_size=8)
                self.assertEqual([int(v[0, 0, 0]) for v in views(path, 'fidpole', args)], [50, 60])
                self.assertEqual([int(v[0, 0, 0]) for v in views(path, 'fidequ', args)], [10, 20, 30, 40])

    def test_fidclip_removes_polar_content_instead_of_rescaling_full_image(self):
        from evaluation.metrics.fid import views
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pano.png'
            image = np.full((40, 80, 3), 100, dtype=np.uint8)
            image[:10] = 0
            image[-10:] = 255
            Image.fromarray(image).save(path)
            cropped = views(path, 'fidclip', SimpleNamespace(clip_crop_fraction=.25))[0]
            self.assertEqual(tuple(cropped.shape), (3, 20, 80))
            self.assertTrue((cropped == 100).all())

    def test_fid_normalizes_and_keeps_reference_and_generated_separate(self):
        import torch
        from evaluation.metrics.fid import compute
        class FakeInception(torch.nn.Module):
            BLOCK_INDEX_BY_DIM = {2048: 3}
            def __init__(self, blocks):
                super().__init__()
            def forward(self, x):
                if not (0 <= x.min() <= x.max() <= 1):
                    raise ValueError('Incorrect Inception input scale')
                return [x.mean((2, 3), keepdim=True)]
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for i, gray in enumerate([0, 255, 64, 128]):
                path = Path(temp) / f'{i}.png'
                Image.new('RGB', (64, 32), (gray, gray, gray)).save(path)
                paths.append(path)
            captured = []
            def distance(real, generated):
                captured.extend([real, generated])
                return 1.0
            fake = SimpleNamespace(InceptionV3=FakeInception)
            with patch.dict('sys.modules', {'pytorch_fid.inception': fake}), \
                    patch('evaluation.metrics.fid.frechet', side_effect=distance), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = compute('fid', [{'image': p} for p in paths[2:]], paths[:2],
                                 SimpleNamespace(device='cpu', batch_size=1))
            self.assertEqual(result['n_real'], 2)
            self.assertEqual(result['n_generated'], 2)
            np.testing.assert_allclose(captured[0].mean, .5)
            np.testing.assert_allclose(captured[1].mean, 96 / 255)

    def test_faed_loads_only_encoder_and_applies_latitude_weighting(self):
        import torch
        from evaluation.metrics.faed import compute
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'models/faed/modules.py'
            source.parent.mkdir(parents=True)
            source.write_text('import torch\nclass Encoder(torch.nn.Module):\n'
                              '    def __init__(self):\n        super().__init__()\n'
                              '        self.scale = torch.nn.Parameter(torch.ones(1))\n'
                              '    def forward(self, x):\n        return x * self.scale\n')
            checkpoint = root / 'faed.ckpt'
            torch.save({'state_dict': {'net.encoder.scale': torch.ones(1),
                                      'net.decoder.unused': torch.zeros(1)}}, checkpoint)
            paths = []
            for i, gray in enumerate([0, 255, 64, 128]):
                path = root / f'{i}.png'
                Image.new('RGB', (64, 32), (gray, gray, gray)).save(path)
                paths.append(path)
            captured = []
            def distance(real, generated):
                captured.extend([real, generated])
                return 1.0
            args = SimpleNamespace(panfusion_root=root, faed_weights=checkpoint,
                                   faed_height=32, device='cpu', batch_size=2,
                                   faed_compare_preprocessing=bool(importlib.util.find_spec('cv2')),
                                   output=root)
            with patch('evaluation.metrics.faed.frechet', side_effect=distance), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = compute('faed', [{'image': p} for p in paths[2:]], paths[:2], args)
            self.assertEqual(result['n_real'], 2)
            np.testing.assert_allclose(captured[0].mean, 0, atol=1e-6)
            feature = captured[1].mean.reshape(3, 32)
            np.testing.assert_allclose(feature[:, [0, -1]], 0, atol=1e-6)
            self.assertLess(feature[0, 16], -.2)
            if args.faed_compare_preprocessing:
                self.assertTrue((root / 'faed_diagnostics.json').is_file())
                self.assertEqual(result['details']['preprocessing_comparison'],
                                 {'pil-bicubic': 1.0, 'panfusion': 1.0})


if __name__ == '__main__':
    unittest.main()
