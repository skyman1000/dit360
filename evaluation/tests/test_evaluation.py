"""CPU-only verification; never downloads pretrained models or evaluates user images."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from evaluation.common import FeatureStats, crop_bounds, inception_score, load_samples
from evaluation.evaluate import blocked_reason, main, parser


class EvaluationTests(unittest.TestCase):
    def test_covariance_matches_numpy_across_uneven_batches(self):
        x = np.random.default_rng(13).normal(size=(19, 6)) + 10000
        s = FeatureStats()
        for batch in (x[:1], x[1:7], x[7:]):
            s.update(batch)
        np.testing.assert_allclose(s.mean, x.mean(0), atol=1e-10)
        np.testing.assert_allclose(s.covariance(), np.cov(x, rowvar=False), atol=1e-10)
        self.assertEqual(s.n, 19)

    def test_is_known_distributions_and_split_validation(self):
        self.assertAlmostEqual(inception_score(np.full((7, 3), 1 / 3), 1)[0], 1)
        self.assertAlmostEqual(inception_score(np.eye(4), 1)[0], 4)
        self.assertAlmostEqual(inception_score(np.eye(4), 4)[0], 1)
        with self.assertRaises(ValueError):
            inception_score(np.eye(4), 5)

    def test_crop_excludes_both_poles(self):
        self.assertEqual(crop_bounds(100, .1), (10, 90))
        for fraction in (0, .5, -.1):
            with self.assertRaises(ValueError):
                crop_bounds(100, fraction)

    def fixture(self, root):
        image = root / 'pano.png'
        Image.new('RGB', (64, 32), (80, 90, 100)).save(image)
        image.with_suffix('.json').write_text(json.dumps({'id': 'pano', 'prompt': 'A room.', 'seed': 0}))
        return image

    def test_manifest_relative_paths_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = self.fixture(root)
            manifest = root / 'list.jsonl'
            line = json.dumps({'image': 'pano.png', 'prompt': 'A room.'}) + '\n'
            manifest.write_text(line)
            self.assertEqual(load_samples(manifest=manifest)[0]['image'], str(image.resolve()))
            manifest.write_text(line * 2)
            with self.assertRaises(ValueError):
                load_samples(manifest=manifest)

    def test_no_reference_and_no_crop_are_not_fabricated(self):
        args = parser().parse_args(['--generated-dir', '.', '--output', 'unused'])
        self.assertIn('reference', blocked_reason('fid', args, [{}] * 2, []))
        args.reference_label = 'synthetic unit test'
        self.assertIn('crop', blocked_reason('fidclip', args, [{}] * 1000, ['x'] * 1000))
        self.assertIn('clip-model', blocked_reason('cs', args, [{'prompt': 'text'}], []))

    def test_dry_run_does_not_import_metric_backends(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / 'images'
            inputs.mkdir()
            self.fixture(inputs)
            output = root / 'report'
            with patch('evaluation.evaluate.importlib.import_module', side_effect=AssertionError('model imported')):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(['--generated-dir', str(inputs), '--output', str(output),
                                 '--metrics', 'all', '--dry-run'])
            self.assertEqual(code, 0)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['results']['fid']['status'], 'skipped')
            self.assertEqual(report['results']['niqe']['status'], 'ready')

    def test_partial_failure_writes_results_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root / 'images'
            inputs.mkdir()
            image = self.fixture(inputs)
            output = root / 'report'
            def compute(name, rows, refs, args):
                if name == 'niqe':
                    raise RuntimeError('Synthetic backend failure')
                return {'value': 12.5, 'n': 1, 'per_image': {str(image): 12.5}}
            backend = SimpleNamespace(compute=compute)
            with patch('evaluation.evaluate.importlib.import_module', return_value=backend):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(['--generated-dir', str(inputs), '--output', str(output),
                                 '--metrics', 'brisque', 'niqe', 'fid'])
            self.assertEqual(code, 2)
            report = json.loads((output / 'summary.json').read_text())
            self.assertEqual(report['results']['brisque']['value'], 12.5)
            self.assertEqual(report['results']['niqe']['status'], 'error')
            self.assertEqual(report['results']['fid']['status'], 'skipped')
            self.assertTrue((output / 'niqe_error.txt').is_file())
            self.assertIn('12.5', (output / 'per_image.csv').read_text())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--generated-dir', str(inputs), '--output', str(output), '--dry-run'])


if __name__ == '__main__':
    unittest.main()
