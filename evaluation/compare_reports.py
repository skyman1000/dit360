"""Combine baseline eight metrics and three regional FIDs, without overwriting either."""
import argparse
import csv
import json
from pathlib import Path

PAPER = {'fid': 42.88, 'fidclip': 41.60, 'fidpole': 50.88, 'fidequ': 24.77,
         'faed': 2.91, 'is': 1.60, 'cs': 34.68, 'qaquality': 4.69,
         'qaaesthetic': 4.19, 'brisque': 10.25, 'niqe': 3.72}
SOURCE = 'https://arxiv.org/html/2510.11712v1#S4.T1'


def comparison_rows(baseline, regions):
    from evaluation.evaluate import HIGHER
    for report in (baseline, regions):
        if not report.get('complete'):
            raise ValueError('Both reports must have completed successfully')
    digest = baseline.get('inputs_manifest_sha256')
    if not digest or digest != regions.get('inputs_manifest_sha256'):
        raise ValueError('Reports use different input inventories; do not merge them blindly')
    rows = []
    for metric, paper in PAPER.items():
        report = regions if metric in ('fidclip', 'fidpole', 'fidequ') else baseline
        result = report['results'].get(metric, {})
        if result.get('status') != 'ok':
            raise ValueError(f'{metric} is not successfully evaluated')
        rows.append({'metric': metric, 'direction': 'higher' if metric in HIGHER else 'lower',
                     'paper': paper, 'current': result['value'], 'current_minus_paper': result['value'] - paper,
                     'source': 'regions' if report is regions else 'baseline',
                     'protocol_status': 'independent; exact DiT360 protocol unverified'})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--regions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from evaluation.common import fingerprint_file
    reports = {k: json.loads(getattr(args, k).read_text()) for k in ('baseline', 'regions')}
    rows = comparison_rows(**reports)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    document = {'paper_source': SOURCE, 'rows': rows,
                'source_reports': {k: {'path': str(getattr(args, k).resolve()),
                                      'sha256': fingerprint_file(getattr(args, k))} for k in reports},
                'region_config': reports['regions']['config'],
                'note': 'Numerical comparison only. Original FAED and IS preserved; diagnostic variants are NOT substituted or selected for closeness to paper.'}
    (args.output / 'comparison.json').write_text(json.dumps(document, indent=2, allow_nan=False))
    for row in rows:
        print(f"{row['metric']:12} paper={row['paper']:.2f} current={row['current']:.4f} delta={row['current_minus_paper']:+.4f}")


if __name__ == '__main__':
    main()
