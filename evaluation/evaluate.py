"""Run from repository root: python -m evaluation.evaluate --help."""
import argparse
import csv
import gc
import importlib
from importlib.metadata import version, PackageNotFoundError
import json
import math
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

from evaluation.common import collect_images, fingerprint_file, load_samples

BACKENDS = {
    'fid': 'fid', 'fidclip': 'fid', 'fidpole': 'fid', 'fidequ': 'fid',
    'faed': 'faed', 'is': 'places_is', 'cs': 'clip_score',
    'qaquality': 'no_reference', 'qaaesthetic': 'no_reference',
    'brisque': 'no_reference', 'niqe': 'no_reference',
}
REFERENCE_METRICS = {'fid', 'fidclip', 'fidpole', 'fidequ', 'faed'}
HIGHER = {'is', 'cs', 'qaquality', 'qaaesthetic'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    generated = p.add_mutually_exclusive_group(required=True)
    generated.add_argument('--generated-dir', help='Recursively read images and matching .json sidecars')
    generated.add_argument('--generated-manifest', help='JSONL with image and (for CS) prompt')
    reference = p.add_mutually_exclusive_group()
    reference.add_argument('--reference-dir', help='Directory of actual reference images, not HF Arrow cache')
    reference.add_argument('--reference-manifest', help='JSONL with image paths')
    p.add_argument('--reference-label', help='Dataset, split and provenance; required for distribution metrics')
    p.add_argument('--output', required=True, help='A NEW output directory (never overwrite an existing report)')
    p.add_argument('--metrics', nargs='+', default=['brisque', 'niqe', 'cs'],
                   help=' '.join(BACKENDS) + '; or all')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--clip-model', help='Explicit CLIP model ID or local path for CS')
    p.add_argument('--clip-model-path', help='Local CLIP snapshot (.bin); overrides loading path for CS only')
    p.add_argument('--qalign-model-path', help='Q-Align local snapshot with .bin weights; only Q-Align uses this path')
    p.add_argument('--clip-crop-fraction', type=float,
                   help='FIDclip: fraction removed from EACH pole; no claimed paper default')
    p.add_argument('--face-size', type=int, help='FIDpole/equ cubemap face width; explicit protocol choice')
    p.add_argument('--panfusion-root', help='PanFusion checkout containing models/faed/modules.py')
    p.add_argument('--faed-weights', help='Local PanFusion faed.ckpt')
    p.add_argument('--faed-height', type=int, help='FAED ERP height, divisible by 32')
    p.add_argument('--faed-preprocess', choices=['pil-bicubic', 'panfusion'], default='pil-bicubic',
                   help='Keep legacy bicubic, or PanFusion reference AREA/generated LINEAR resizing')
    p.add_argument('--faed-compare-preprocessing', action='store_true',
                   help='Evaluate both resize paths with identical encoder/inputs; save diagnostics')
    p.add_argument('--places-weights', help='Local Places365 ResNet checkpoint')
    p.add_argument('--places-arch', choices=['resnet18', 'resnet50'])
    p.add_argument('--is-splits', type=int, help='Number of splits for Places365-IS')
    p.add_argument('--is-diagnostics', action='store_true',
                   help='Save probabilities, entropy and ordered/shuffled/grouped IS diagnostics')
    p.add_argument('--dry-run', action='store_true', help='Validate input/config only; no GPU or weight download')
    return p


def blocked_reason(name, args, rows, references):
    if name in ('qaquality', 'qaaesthetic') and args.qalign_model_path:
        from evaluation.metrics.qalign_local import validate_snapshot
        try:
            validate_snapshot(args.qalign_model_path)
        except (ValueError, OSError, KeyError) as error:
            return str(error)
    if name in REFERENCE_METRICS:
        if len(references) < 2 or len(rows) < 2:
            return 'Need at least 2 real and 2 generated images; provide --reference-dir/manifest'
        if not args.reference_label:
            return 'Provide --reference-label with dataset/split provenance'
    if name == 'fidclip' and (args.clip_crop_fraction is None or not 0 < args.clip_crop_fraction < .5):
        return 'Set --clip-crop-fraction in (0,0.5); no verified paper crop value is available'
    if name in ('fidpole', 'fidequ') and (args.face_size is None or args.face_size < 2):
        return 'Set --face-size >=2'
    if name == 'faed':
        if not args.panfusion_root or not (Path(args.panfusion_root) / 'models/faed/modules.py').is_file():
            return 'Provide --panfusion-root pointing to the public PanFusion checkout'
        if not args.faed_weights or not Path(args.faed_weights).is_file():
            return 'Provide local --faed-weights (PanFusion faed.ckpt)'
        if not args.faed_height or args.faed_height % 32 or args.faed_height < 32:
            return 'Set --faed-height to a positive multiple of 32'
    if name == 'is':
        if not args.places_weights or not Path(args.places_weights).is_file() or not args.places_arch:
            return 'Provide --places-weights and --places-arch; no Inception-v3 substitution'
        if args.is_splits is None or not 1 <= args.is_splits <= len(rows):
            return 'Set --is-splits between 1 and generated image count'
    if name == 'cs':
        if args.clip_model_path:
            from evaluation.metrics.clip_score import validate_local_clip
            try:
                validate_local_clip(args.clip_model_path)
            except (ValueError, OSError) as error:
                return str(error)
        if not args.clip_model and not args.clip_model_path:
            return 'Set --clip-model explicitly (DiT360 CLIP variant is unverified)'
        missing = [r['image'] for r in rows if not isinstance(r.get('prompt'), str) or not r['prompt'].strip()]
        if missing:
            return f'{len(missing)} images lack prompt metadata, e.g. {missing[0]}'
    return None


def write_reports(output, report, per_image):
    # Replace only files created inside this run's newly-created directory.
    tmp = output / 'summary.json.tmp'
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(output / 'summary.json')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['metric', 'direction', 'status', 'value', 'std', 'n', 'n_real', 'n_generated', 'reason']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, result in report['results'].items():
            writer.writerow({'metric': name, 'direction': 'higher' if name in HIGHER else 'lower', **result})
    with (output / 'per_image.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['image', 'id', 'seed', 'prompt'] + report['selected_metrics']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(per_image.values())


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    selected = list(BACKENDS) if args.metrics == ['all'] else list(dict.fromkeys(args.metrics))
    unknown = set(selected) - set(BACKENDS)
    if unknown:
        p.error(f'Unknown metrics: {sorted(unknown)}')
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    try:
        rows = load_samples(args.generated_dir, args.generated_manifest)
        references = (collect_images(args.reference_dir) if args.reference_dir else
                      [Path(r['image']) for r in load_samples(manifest=args.reference_manifest)]
                      if args.reference_manifest else [])
        if set(r['image'] for r in rows) & set(map(str, references)):
            raise ValueError('Generated and reference inputs overlap; use distinct image sets')
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as e:
        p.error(str(e))

    per_image = {r['image']: dict(r) for r in rows}
    packages = {}
    for package in ['torch', 'torchvision', 'numpy', 'scipy', 'Pillow', 'pytorch-fid',
                    'torchmetrics', 'torch-fidelity', 'transformers', 'pyiqa', 'py360convert',
                    'opencv-python', 'opencv-python-headless']:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    report = {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'config': vars(args),
        'protocol': 'independent implementation; NOT verified identical to DiT360 paper protocol',
        'n_generated_images': len(rows), 'n_reference_images': len(references),
        'selected_metrics': selected, 'package_versions': packages,
        'code_sha256': {str(path.relative_to(Path(__file__).parent)): fingerprint_file(path)
                        for path in sorted(Path(__file__).parent.rglob('*.py'))},
        'warnings': [], 'results': {},
    }
    if len(rows) < 1000 or (references and len(references) < 1000):
        report['warnings'].append('Small sample estimates can be unstable. This <1000 reminder is not a paper requirement; no samples are added, removed or blocked.')
    report['warnings'].append('Metrics do not establish causal reasoning or correct 3D geometry; retain visual/global consistency checks.')
    # Freeze the exact ordered paths, original generation metadata and file stat inventory.
    with (output / 'inputs.jsonl').open('w', encoding='utf-8') as f:
        for kind, items in [('generated', rows), ('reference', [{'image': str(p)} for p in references])]:
            for row in items:
                stat = Path(row['image']).stat()
                f.write(json.dumps({**row, 'set': kind, 'size_bytes': stat.st_size,
                                    'mtime_ns': stat.st_mtime_ns}, ensure_ascii=False) + '\n')
    report['inputs_manifest_sha256'] = fingerprint_file(output / 'inputs.jsonl')
    for name in selected:
        reason = blocked_reason(name, args, rows, references)
        report['results'][name] = ({'status': 'skipped', 'reason': reason} if reason else
                                   {'status': 'ready' if args.dry_run else 'pending'})
    write_reports(output, report, per_image)
    print(f'Generated: {len(rows)}; reference: {len(references)}; report: {output}', flush=True)
    if args.dry_run:
        for name, result in report['results'].items():
            print(f'{name}: {result}', flush=True)
        print('Dry run checks inputs/config only; model dependencies and weights were not executed.', flush=True)
        return 0

    interrupted = False
    for name in selected:
        entry = report['results'][name]
        if entry['status'] == 'skipped':
            print(f'[{name}] SKIPPED: {entry["reason"]}', flush=True)
            continue
        print(f'[{name}] Loading metric/model (first use may download weights)...', flush=True)
        start = time.monotonic()
        try:
            module = importlib.import_module('evaluation.metrics.' + BACKENDS[name])
            result = module.compute(name, rows, references, args)
            if not math.isfinite(result['value']):
                raise ValueError('Non-finite aggregate value')
            values = result.pop('per_image', {})
            for image, score in values.items():
                per_image[image][name] = score
            entry.update(result, status='ok')
            print(f'[{name}] {result["value"]:.6f}', flush=True)
        except KeyboardInterrupt:
            entry.update(status='interrupted', reason='User interrupted this metric')
            interrupted = True
        except Exception as error:
            entry.update(status='error', reason=f'{type(error).__name__}: {error}')
            (output / f'{name}_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            print(f'[{name}] ERROR: {error}', flush=True)
        finally:
            entry['seconds'] = round(time.monotonic() - start, 3)
            gc.collect()
            # Do not import torch for skipped metrics or dry runs.
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_reports(output, report, per_image)
        if interrupted:
            break
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['complete'] = all(r['status'] == 'ok' for r in report['results'].values())
    write_reports(output, report, per_image)
    print(f'Saved: {output / "summary.csv"}', flush=True)
    return 130 if interrupted else 0 if report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
