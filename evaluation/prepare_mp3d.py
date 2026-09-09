"""Prepare the public predecessor split, NOT a verified DiT360 evaluation split.

Read only selected ZIP members (ZIP CRC is checked on read). Use PanFusion's
bundled py360convert and its Matterport skybox orientation convention:
https://github.com/chengzhag/PanFusion/blob/main/utils/pano.py
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile


def stitched_caption(root, scene, view):
    matches = sorted(root.glob(f'**/{scene}/blip3_stitched/{view}.txt'))
    if len(matches) != 1:
        raise ValueError(f'Expected one full-panorama caption for {scene}/{view}; found {len(matches)} under {root}')
    caption = matches[0].read_text().strip()
    if not caption:
        raise ValueError(f'Empty full-panorama caption: {matches[0]}')
    prefix = 'This is a panorama.'
    prompt = caption if caption.lower().startswith(prefix.lower()) else prefix + ' ' + caption
    return caption, prompt, str(matches[0].resolve())


def select_per_scene(entries, count, seed):
    import numpy as np
    groups = {}
    for index, row in enumerate(entries):
        value = row[0].decode() if isinstance(row[0], bytes) else str(row[0])
        groups.setdefault(value.split('/')[0], []).append(index)
    rng = np.random.default_rng(seed)
    selected = []
    for scene in sorted(groups):
        if len(groups[scene]) < count:
            raise ValueError(f'{scene} has fewer than {count} views')
        selected.extend(rng.choice(groups[scene], count, replace=False).tolist())
    return entries[sorted(selected)]


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--panfusion-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, help='Diagnostic subset; use a separate output directory')
    p.add_argument('--per-scene', type=int, help='Random views per building for a cross-scene diagnostic')
    p.add_argument('--selection-seed', type=int, default=0)
    p.add_argument('--caption-mode', choices=['multiview', 'stitched'], default='multiview')
    p.add_argument('--caption-root', type=Path, help='Extracted panorama caption package root; required for stitched mode')
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if args.per_scene is not None and (args.per_scene < 1 or args.limit is not None):
        p.error('--per-scene must be positive and cannot be combined with --limit')
    if args.caption_mode == 'stitched' and (args.caption_root is None or not args.caption_root.is_dir()):
        p.error('stitched mode requires an existing --caption-root; no fallback to multiview captions')
    root = args.assets.resolve()
    meta = root / 'Matterport3D_metadata/mp3d_skybox'
    split = meta / 'test.npy'
    entries = np.load(split, allow_pickle=False)
    if entries.ndim != 2 or entries.shape[1] != 6:
        raise ValueError(f'Expected six face paths per record, got {entries.shape}')
    if args.limit:
        entries = entries[:args.limit]
    if args.per_scene:
        entries = select_per_scene(entries, args.per_scene, args.selection_seed)
    sys.path.insert(0, str(args.panfusion_root.resolve()))
    from external import py360convert
    converter = Path(py360convert.__file__).resolve().parent
    converter_hash = hashlib.sha256()
    for file in sorted(converter.rglob('*.py')):
        converter_hash.update(str(file.relative_to(converter)).encode())
        converter_hash.update(file.read_bytes())
    rows = []
    archives = {}
    try:
        for entry in entries:
            paths = [str(x.decode() if isinstance(x, bytes) else x) for x in entry]
            match = re.fullmatch(r'([A-Za-z0-9]+)/matterport_skybox_images/([A-Za-z0-9]+)_skybox0_sami.jpg', paths[0])
            if not match:
                raise ValueError(f'Unexpected split path: {paths[0]}')
            scene, view = match.groups()
            expected = [f'{scene}/matterport_skybox_images/{view}_skybox{i}_sami.jpg' for i in range(6)]
            if paths != expected:
                raise ValueError(f'Unexpected face ordering: {paths}')
            if scene not in archives:
                archive = zipfile.ZipFile(root / f'Matterport3D_raw/v1/scans/{scene}/matterport_skybox_images.zip')
                archives[scene] = (archive, {})
                index = archives[scene][1]
                for name in archive.namelist():
                    normalized = str(Path(name))
                    if normalized in index:
                        raise ValueError(f'Duplicate ZIP member: {normalized}')
                    index[normalized] = name
            archive, index = archives[scene]
            for name in paths:
                if name not in index:
                    raise ValueError(f'Missing ZIP member: {name}')
            extra = {}
            if args.caption_mode == 'stitched':
                caption, prompt, source = stitched_caption(args.caption_root.resolve(), scene, view)
                captions = [caption]
                extra = dict(caption_mode='stitched', caption_file=source)
            else:
                captions = []
                for angle in range(0, 360, 45):
                    file = meta / scene / 'blip3' / f'{view}_{angle}.txt'
                    caption = file.read_text().strip()
                    if not caption:
                        raise ValueError(f'Empty caption: {file}')
                    captions.append(caption)
                prompt = 'This is a panorama. ' + '. '.join(captions)
            rows.append(dict(id=f'{scene}_{view}', scene_id=scene, view_id=view,
                             prompt=prompt, source_captions=captions, skybox_paths=paths, **extra))
        if not rows or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Empty split or duplicate views')
        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=True)
        config = dict(protocol='PanFusion/MVDiffusion candidate; exact DiT360 split/prompts unverified',
                      count=len(rows), split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                      prompt_rule='8 blip3 captions in yaw order 0,45,...315 joined with . ; panorama prefix',
                      rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                      converter_sha256=converter_hash.hexdigest(), width=2048, height=1024,
                      interpolation='bilinear', limit=args.limit,
                      archive_stats={s: [Path(z.filename).stat().st_size, Path(z.filename).stat().st_mtime_ns]
                                     for s, (z, _) in archives.items()})
        protocol = out / 'protocol.json'
        if args.caption_mode == 'stitched':
            config['prompt_rule'] = 'Single provided blip3_stitched caption; add panorama prefix only if absent'
            config['caption_mode'] = 'stitched'
        if args.per_scene:
            config['selection'] = dict(per_scene=args.per_scene, seed=args.selection_seed)
        if protocol.exists():
            if json.loads(protocol.read_text()) != config:
                raise ValueError('Existing preparation configuration differs; use a NEW output directory')
        elif any(out.iterdir()):
            raise ValueError('Output is nonempty without protocol.json; use a NEW directory')
        else:
            write_json(protocol, config)
        (out / 'reference').mkdir(exist_ok=True)
        for i, row in enumerate(rows, 1):
            dest = out / 'reference' / (row['id'] + '.png')
            if dest.exists():
                with Image.open(dest) as image:
                    if image.size != (2048, 1024):
                        raise ValueError(f'Wrong dimensions: {dest}')
                    image.verify()
                print(f'[prepare {i}/{len(rows)}] skip {row["id"]}', flush=True)
                continue
            archive, index = archives[row['scene_id']]
            faces = {}
            for key, name in zip(['U', 'L', 'F', 'R', 'B', 'D'], row['skybox_paths']):
                with Image.open(io.BytesIO(archive.read(index[name]))) as image:
                    faces[key] = np.array(image.convert('RGB'))
            faces['R'] = np.flip(faces['R'], 1)
            faces['B'] = np.flip(faces['B'], 1)
            faces['U'] = np.rot90(np.flip(faces['U'], 0), 1)
            faces['D'] = np.rot90(faces['D'], 1)
            cube = py360convert.cube_dict2h(faces)
            pano = py360convert.c2e(cube, 1024, 2048, mode='bilinear', cube_format='horizon')
            tmp = dest.with_suffix('.tmp')
            Image.fromarray(pano.astype(np.uint8)).save(tmp, format='PNG')
            tmp.replace(dest)
            print(f'[prepare {i}/{len(rows)}] saved {row["id"]}', flush=True)
        for filename, records in [('prompts.jsonl', rows), ('reference.jsonl',
                [{**r, 'image': f'reference/{r["id"]}.png'} for r in rows])]:
            tmp = out / (filename + '.tmp')
            tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
            tmp.replace(out / filename)
        print(f'Complete: {len(rows)} views; manifests in {out}', flush=True)
    finally:
        for archive, _ in archives.values():
            archive.close()


if __name__ == '__main__':
    main()
