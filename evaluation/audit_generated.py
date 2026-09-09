"""CPU-only prompt, seed and decoded-pixel duplicate audit; no models."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from evaluation.common import load_samples, open_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Audit output already exists; choose a new filename')
    rows = load_samples(manifest=args.manifest)
    groups = {key: defaultdict(list) for key in ('prompt', 'prompt_seed', 'pixels')}
    for i, row in enumerate(rows, 1):
        label = row.get('id', row['image'])
        groups['prompt'][row.get('prompt', '')].append(label)
        groups['prompt_seed'][json.dumps([row.get('prompt'), row.get('seed')])].append(label)
        im = open_rgb(row['image'])
        digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        groups['pixels'][digest].append(label)
        print(f'[audit {i}/{len(rows)}] {label}', flush=True)
    report = dict(count=len(rows), unique_pixel_images=len(groups['pixels']),
                  duplicate_groups={k: [v for v in g.values() if len(v) > 1] for k, g in groups.items()},
                  note='Pixel equality detects exact duplicates, not perceptual similarity; no images removed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
