"""Check complete, matching benchmark manifests before invoking the evaluator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Empty manifest or duplicate IDs: {path}')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--generated', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args, extra = p.parse_known_args()
    forbidden = ('--generated-dir', '--generated-manifest', '--reference-dir',
                 '--reference-manifest', '--reference-label')
    if any(x.split('=')[0] in forbidden for x in extra):
        p.error('Input manifests and reference label are managed by this wrapper')
    ref = args.prepared.resolve() / 'reference.jsonl'
    gen = args.generated.resolve() / 'generated.jsonl'
    references, generated = read(ref), read(gen)
    protocol = json.loads((args.prepared / 'protocol.json').read_text())
    if len(references) != protocol['count']:
        raise ValueError('Reference count differs from preparation protocol')
    if [r['id'] for r in references] != [r['id'] for r in generated]:
        raise ValueError('Generated IDs/count/order do not match the reference split')
    if any(a['prompt'] != b['prompt'] for a, b in zip(references, generated)):
        raise ValueError('Generated prompts differ from reference manifest')
    if not any(x == '--metrics' or x.startswith('--metrics=') for x in extra):
        extra += ['--metrics', 'fid', 'brisque', 'niqe', 'cs', 'qaquality', 'qaaesthetic']
    cmd = [sys.executable, '-u', '-m', 'evaluation.evaluate',
           '--generated-manifest', str(gen), '--reference-manifest', str(ref),
           '--reference-label', protocol['protocol'], '--output', str(args.output.resolve()), *extra]
    print(f'Validated matching manifests: {len(generated)} views', flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=Path(__file__).resolve().parents[1]))


if __name__ == '__main__':
    main()
