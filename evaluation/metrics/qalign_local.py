"""Scope Q-Align's hard-coded HF loaders to a local snapshot, without env changes."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace


def validate_snapshot(directory):
    root = Path(directory).expanduser().resolve()
    required = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                'tokenizer.model', 'pytorch_model.bin.index.json']
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete Q-Align snapshot {root}: missing {missing}')
    index = json.loads((root / 'pytorch_model.bin.index.json').read_text())
    shards = set(index['weight_map'].values())
    if not shards:
        raise ValueError('Q-Align weight index is empty')
    for shard in shards:
        path = root / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing/empty Q-Align weight shard: {path}')
    return str(root)


def local_loader(loader, root, *, weights=False):
    def from_pretrained(_original_id, *args, **kwargs):
        kwargs['local_files_only'] = True
        kwargs.pop('revision', None)
        kwargs.pop('cache_dir', None)
        if weights:
            kwargs['use_safetensors'] = False
        return loader(root, *args, **kwargs)
    return SimpleNamespace(from_pretrained=from_pretrained)


@contextmanager
def local_qalign_loaders(directory):
    root = validate_snapshot(directory)
    try:
        from pyiqa.archs import qalign_arch
        from pyiqa.archs.q_align import modeling_mplug_owl2
        model = qalign_arch.MPLUGOwl2LlamaForCausalLM
    except (ImportError, AttributeError) as error:
        raise RuntimeError('Local Q-Align mode needs pyiqa with its bundled q_align implementation '
                           '(the installed pyiqa 0.1.15 supports it); no remote-code fallback.') from error
    # The model constructor also hard-codes tokenizer and processor IDs. Redirect
    # those module-local names as well. Other Transformers/HF loaders stay intact.
    targets = [
        (qalign_arch, 'MPLUGOwl2LlamaForCausalLM', local_loader(model.from_pretrained, root, weights=True)),
        (qalign_arch, 'CLIPImageProcessor', local_loader(qalign_arch.CLIPImageProcessor.from_pretrained, root)),
        (modeling_mplug_owl2, 'AutoTokenizer', local_loader(modeling_mplug_owl2.AutoTokenizer.from_pretrained, root)),
        (modeling_mplug_owl2, 'CLIPImageProcessor', local_loader(modeling_mplug_owl2.CLIPImageProcessor.from_pretrained, root)),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in targets]
    try:
        for module, name, replacement in targets:
            setattr(module, name, replacement)
        yield root
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def create_local_metric(directory, device):
    import pyiqa
    with local_qalign_loaders(directory):
        return pyiqa.create_metric('qalign', device=device).eval()
