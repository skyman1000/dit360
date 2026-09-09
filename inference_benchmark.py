"""Resumable candidate MP3D benchmark generation; upstream inference.py unchanged."""
import argparse
import hashlib
import json
from pathlib import Path


def sample_seed(base_seed, sample_id, mode):
    if mode == 'shared':
        return base_seed
    # Stable across machines, ordering, subsets and resumed runs (unlike hash()).
    digest = hashlib.sha256(f'{base_seed}:{sample_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % (2 ** 63)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seed-mode', choices=['shared', 'per-id'], default='shared',
                   help='shared preserves old runs; per-id derives reproducible noise per viewpoint')
    args = p.parse_args()
    rows = [json.loads(x) for x in args.prompts.read_text().splitlines() if x.strip()]
    import re
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate input IDs')
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']) or not row['prompt'].strip():
            raise ValueError('Unsafe ID or empty prompt')
    seeds = [sample_seed(args.seed, row['id'], args.seed_mode) for row in rows]
    if args.seed_mode == 'per-id' and len(set(seeds)) != len(seeds):
        raise ValueError('Derived seed collision; choose another base seed')
    repeated = len(rows) - len({r['prompt'] for r in rows})
    print(f'Inputs: {len(rows)}; repeated prompt occurrences: {repeated}; seed mode: {args.seed_mode}', flush=True)
    if repeated and args.seed_mode == 'shared':
        print('WARNING: identical prompts with shared seeds can generate identical images.', flush=True)
    import torch
    from huggingface_hub import snapshot_download
    from src.pipeline import DiT360Pipeline
    model_id = 'black-forest-labs/FLUX.1-dev'
    lora_id = 'Insta360-Research/DiT360-Panorama-Image-Generation'
    model_path = snapshot_download(model_id, local_files_only=True)
    lora_path = snapshot_download(lora_id, local_files_only=True)
    config = dict(model=model_id, lora=lora_id, model_snapshot=model_path,
                  lora_snapshot=lora_path, width=2048, height=1024, steps=28,
                  guidance_scale=3.0, seed=args.seed, dtype='float16',
                  prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(), count=len(rows))
    if args.seed_mode == 'per-id':
        config.update(seed_mode='per-id', base_seed=args.seed,
                      seed_derivation='sha256(utf8(base_seed:sample_id)) first8 big-endian modulo 2**63')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = out / 'generation_config.json'
    if cfg.exists():
        if json.loads(cfg.read_text()) != config:
            raise ValueError('Configuration changed; use a NEW output directory')
    elif any(out.iterdir()):
        raise ValueError('Nonempty output without generation_config.json')
    else:
        cfg.write_text(json.dumps(config, indent=2))
    pipe = None
    generated = []
    from PIL import Image
    for i, row in enumerate(rows, 1):
        actual_seed = seeds[i - 1]
        dest = out / f'{row["id"]}_seed{actual_seed}.png'
        sidecar = dest.with_suffix('.json')
        metadata = {**row, **config, 'seed': actual_seed}
        if dest.exists() and sidecar.exists():
            if json.loads(sidecar.read_text()) != metadata:
                raise ValueError(f'Metadata mismatch: {sidecar}')
            with Image.open(dest) as im:
                if im.size != (2048, 1024):
                    raise ValueError(f'Wrong image dimensions: {dest}')
                im.verify()
            print(f'[{i}/{len(rows)}] skip {row["id"]}', flush=True)
        else:
            if pipe is None:
                print('Loading cached FLUX and DiT360 LoRA...', flush=True)
                pipe = DiT360Pipeline.from_pretrained(model_path, torch_dtype=torch.float16,
                                                     local_files_only=True).to('cuda:0')
                pipe.load_lora_weights(lora_path, weight_name='adapter_model.safetensors', local_files_only=True)
                pipe.set_progress_bar_config(disable=False)
            print(f'[{i}/{len(rows)}] generating {row["id"]}', flush=True)
            image = pipe(row['prompt'], width=2048, height=1024, num_inference_steps=28,
                         guidance_scale=3.0,
                         generator=torch.Generator(device='cuda:0').manual_seed(actual_seed)).images[0]
            tmp = dest.with_suffix('.tmp')
            image.save(tmp, format='PNG')
            tmp.replace(dest)
            tmp = sidecar.with_suffix('.tmp')
            tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
            tmp.replace(sidecar)
        generated.append({**metadata, 'image': dest.name})
    # Publish the evaluation manifest ONLY after every requested image is complete.
    tmp = out / 'generated.jsonl.tmp'
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in generated))
    tmp.replace(out / 'generated.jsonl')
    print(f'Complete: {len(generated)} images; {out / "generated.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
