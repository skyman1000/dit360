"""CS = max(100 * cosine(CLIP(image), CLIP(text)), 0), per TorchMetrics."""
from evaluation.common import image_tensor, per_image_result, progress
from pathlib import Path


def validate_local_clip(directory):
    root = Path(directory).expanduser().resolve()
    for name in ['config.json', 'preprocessor_config.json', 'tokenizer_config.json',
                 'vocab.json', 'merges.txt', 'pytorch_model.bin']:
        if not (root / name).is_file() or not (root / name).stat().st_size:
            raise ValueError(f'Missing local CLIP file: {root / name}')
    return str(root)


def local_clip_factory(directory):
    root = validate_local_clip(directory)
    def load():
        from transformers import CLIPModel, CLIPProcessor
        return (CLIPModel.from_pretrained(root, local_files_only=True, use_safetensors=False),
                CLIPProcessor.from_pretrained(root, local_files_only=True, use_fast=False))
    return load


def encode_text(processor, prompt, max_length):
    """Truncate content BEFORE adding special tokens; never slice away EOS."""
    import torch
    tokenizer = processor.tokenizer
    # This untruncated pass only counts tokens; it is never sent to the model.
    raw = tokenizer(prompt, add_special_tokens=False, truncation=False, verbose=False)['input_ids']
    encoded = processor(text=[prompt], padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
    ids = encoded['input_ids']
    mask = encoded['attention_mask']
    last = mask.sum(-1) - 1
    if ids.shape[-1] > max_length or not torch.all(
            ids[torch.arange(ids.shape[0]), last] == tokenizer.eos_token_id):
        raise ValueError('CLIP text must end with EOS within the context window')
    return encoded, {'original_tokens': len(raw) + tokenizer.num_special_tokens_to_add(),
                     'encoded_tokens': int(mask.sum()),
                     'truncated': len(raw) + tokenizer.num_special_tokens_to_add() > max_length,
                     'effective_text': tokenizer.decode(ids[0], skip_special_tokens=True)}


def cosine_score(image_features, text_features):
    import torch
    for features in (image_features, text_features):
        if not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
            raise ValueError('Invalid CLIP features')
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return (100 * (image_features * text_features).sum(-1)).clamp_min(0)


def compute(name, rows, references, args):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    local_path = getattr(args, 'clip_model_path', None)
    if local_path:
        model, processor = local_clip_factory(local_path)()
    else:
        model = CLIPModel.from_pretrained(args.clip_model)
        processor = CLIPProcessor.from_pretrained(args.clip_model, use_fast=False)
    model = model.to(args.device).eval()
    max_length = model.config.text_config.max_position_embeddings
    values, audit = {}, {}
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            text, info = encode_text(processor, row['prompt'], max_length)
            pixels = processor(images=[image_tensor(row['image'])], return_tensors='pt', padding=True)
            image_features = model.get_image_features(pixels['pixel_values'].to(args.device))
            text_features = model.get_text_features(input_ids=text['input_ids'].to(args.device),
                                                   attention_mask=text['attention_mask'].to(args.device))
            values[row['image']] = float(cosine_score(image_features, text_features).item())
            audit[row['image']] = info
            progress(name, i, len(rows))
    return per_image_result(values, {
        'backend': 'transformers CLIP; tokenizer-level truncation with EOS validation',
        'protocol_version': 'cs_eos_v2', 'model': args.clip_model,
        'scale': 'max(100*cosine,0)', 'local_snapshot': local_path,
        'local_files_only': bool(local_path),
        'input': 'full ERP supplied; checkpoint image processor resize/center-crop unchanged',
        'image_processor': processor.image_processor.to_dict(),
        'prompt': 'exact input prompt; tokenizer truncates content and preserves EOS',
        'max_text_tokens': max_length,
        'truncated_count': sum(r['truncated'] for r in audit.values()),
        'text_audit': audit})
