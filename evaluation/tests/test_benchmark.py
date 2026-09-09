"""Protect comparison integrity without loading any models."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import benchmark


class BenchmarkTests(unittest.TestCase):
    def invoke(self, generated, prompt='scene'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'reference.jsonl').write_text(json.dumps({'id': 'a', 'prompt': prompt}) + '\n')
            (root / 'generated.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in generated))
            (root / 'protocol.json').write_text(json.dumps({'count': 1, 'protocol': 'candidate'}))
            args = ['benchmark', '--prepared', directory, '--generated', directory,
                    '--output', str(root / 'report')]
            with patch('sys.argv', args), patch.object(benchmark.subprocess, 'call') as call:
                call.return_value = 0
                try:
                    benchmark.main()
                except SystemExit as result:
                    self.assertEqual(result.code, 0)
                return call.call_args

    def test_wrong_view_rejected(self):
        with self.assertRaisesRegex(ValueError, 'IDs/count/order'):
            self.invoke([{'id': 'b', 'prompt': 'scene'}])

    def test_wrong_prompt_rejected(self):
        with self.assertRaisesRegex(ValueError, 'prompts differ'):
            self.invoke([{'id': 'a', 'prompt': 'another scene'}])

    def test_matching_inputs_invoke_evaluator(self):
        call = self.invoke([{'id': 'a', 'prompt': 'scene'}])
        self.assertIn('evaluation.evaluate', call.args[0])
        self.assertIn('--reference-manifest', call.args[0])


if __name__ == '__main__':
    unittest.main()
