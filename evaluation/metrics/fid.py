"""Inception FID on ERP, clipped ERP, or grouped cubemap faces."""
from evaluation.common import (FeatureStats, crop_bounds, frechet, image_tensor,
                               open_rgb, progress)


def views(path, name, args):
    if name == 'fid':
        return [image_tensor(path)]
    if name == 'fidclip':
        image = image_tensor(path)
        top, bottom = crop_bounds(image.shape[1], args.clip_crop_fraction)
        return [image[:, top:bottom, :]]
    import numpy as np
    import torch
    import py360convert
    image = np.array(open_rgb(path))
    if image.shape[1] != image.shape[0] * 2:
        raise ValueError(f'Cubemap requires a 2:1 ERP image: {path}')
    faces = py360convert.e2c(image, face_w=args.face_size,
                           mode='bilinear', cube_format='dict')
    keys = face_keys(name)
    return [torch.from_numpy(np.ascontiguousarray(faces[k])).permute(2, 0, 1)
            for k in keys]


def face_keys(name):
    if name == 'fidpole':
        return ('U', 'D')
    if name == 'fidequ':
        return ('F', 'R', 'B', 'L')
    raise ValueError(f'Unknown cubemap FID: {name}')


def compute(name, rows, references, args):
    import torch
    from pytorch_fid.inception import InceptionV3
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval().to(args.device)
    distributions = []
    with torch.inference_mode():
        for label, paths in [('real', references), ('generated', [r['image'] for r in rows])]:
            stats = FeatureStats()
            for i, path in enumerate(paths, 1):
                # Faces of one panorama share a size; ERP images may differ in size.
                x = torch.stack(views(path, name, args)).to(args.device).float() / 255
                for batch in x.split(args.batch_size):
                    features = model(batch)[0].flatten(1).cpu().numpy()
                    stats.update(features)
                progress(f'{name}/{label}', i, len(paths))
            distributions.append(stats)
    real, generated = distributions
    cube = name in ('fidpole', 'fidequ')
    return {'value': frechet(real, generated), 'n_real': real.n, 'n_generated': generated.n,
            'details': {'backend': 'pytorch-fid InceptionV3 2048',
                        'input': name, 'resize': 'backend bilinear 299x299',
                        'crop_fraction_each_pole': args.clip_crop_fraction if name == 'fidclip' else None,
                        'face_size': args.face_size if name in ('fidpole', 'fidequ') else None,
                        'projection': 'py360convert e2c bilinear' if name in ('fidpole', 'fidequ') else None,
                        'faces': list(face_keys(name)) if cube else None,
                        'n_real_panoramas': len(references), 'n_generated_panoramas': len(rows),
                        'views_per_panorama': len(face_keys(name)) if cube else 1,
                        'pooling': 'all selected faces pooled; NOT mean of per-face FIDs' if cube else 'one feature per panorama',
                        'protocol_status': 'independent; exact DiT360 crop/projection/backend unverified',
                        'crop_rounding': 'floor(height*fraction) removed from each end' if name == 'fidclip' else None,
                        'region_definition_source': 'SMGD CVPR2025 section 3.4 (cubemap groups)' if cube else None}}
