import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evaluation.metrics.qalign_local import validate_snapshot, local_loader, local_qalign_loaders


class LocalQAlignTests(unittest.TestCase):
    def snapshot(self, root):
        for name in ['config.json', 'preprocessor_config.json', 'tokenizer_config.json']:
            (root / name).write_text('{}')
        (root / 'tokenizer.model').write_bytes(b'test')
        (root / 'weights.bin').write_bytes(b'test')
        (root / 'pytorch_model.bin.index.json').write_text(json.dumps({'weight_map': {'x': 'weights.bin'}}))

    def test_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.snapshot(root)
            self.assertEqual(validate_snapshot(root), str(root))
            (root / 'weights.bin').unlink()
            with self.assertRaisesRegex(ValueError, 'weight shard'):
                validate_snapshot(root)

    def test_model_loader_forces_local_bin_without_mutating_environment(self):
        import os
        before = dict(os.environ)
        calls = []
        proxy = local_loader(lambda *a, **kw: calls.append((a, kw)), '/local/model', weights=True)
        proxy.from_pretrained('q-future/one-align', cache_dir='/other', revision='main')
        self.assertEqual(calls, [(('/local/model',), {'local_files_only': True, 'use_safetensors': False})])
        self.assertEqual(dict(os.environ), before)

    def test_all_nested_loaders_redirect_and_restore_after_error(self):
        calls = []
        loader = SimpleNamespace(from_pretrained=lambda *a, **kw: calls.append((a, kw)))
        arch = SimpleNamespace(MPLUGOwl2LlamaForCausalLM=loader, CLIPImageProcessor=loader)
        model = SimpleNamespace(AutoTokenizer=loader, CLIPImageProcessor=loader)
        modules = {'pyiqa': SimpleNamespace(),
                   'pyiqa.archs': SimpleNamespace(qalign_arch=arch),
                   'pyiqa.archs.q_align': SimpleNamespace(modeling_mplug_owl2=model)}
        with tempfile.TemporaryDirectory() as temp:
            self.snapshot(Path(temp))
            with patch.dict('sys.modules', modules):
                with self.assertRaisesRegex(RuntimeError, 'test error'):
                    with local_qalign_loaders(temp):
                        arch.MPLUGOwl2LlamaForCausalLM.from_pretrained('remote')
                        arch.CLIPImageProcessor.from_pretrained('remote')
                        model.AutoTokenizer.from_pretrained('remote')
                        model.CLIPImageProcessor.from_pretrained('remote')
                        raise RuntimeError('test error')
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(c[0] == (temp,) and c[1]['local_files_only'] for c in calls))
            self.assertIs(arch.MPLUGOwl2LlamaForCausalLM, loader)
            self.assertIs(arch.CLIPImageProcessor, loader)
            self.assertIs(model.AutoTokenizer, loader)
            self.assertIs(model.CLIPImageProcessor, loader)
