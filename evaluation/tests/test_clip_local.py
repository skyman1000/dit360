import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evaluation.metrics.clip_score import local_clip_factory, validate_local_clip


class LocalClipTests(unittest.TestCase):
    def test_incomplete_cache_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                validate_local_clip(temp)

    def test_both_model_and_processor_are_local_and_no_fast_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ['config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                         'vocab.json', 'merges.txt', 'pytorch_model.bin']:
                (Path(temp) / name).write_text('{}')
            calls = []
            fake = SimpleNamespace(from_pretrained=lambda *a, **kw: calls.append((a, kw)))
            with patch.dict('sys.modules', {'transformers': SimpleNamespace(CLIPModel=fake, CLIPProcessor=fake)}):
                local_clip_factory(temp)()
            self.assertEqual(calls, [((temp,), {'local_files_only': True, 'use_safetensors': False}),
                                     ((temp,), {'local_files_only': True, 'use_fast': False})])
