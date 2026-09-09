"""Adapt PanFusion's public encoder and latitude-weighted FAED features.

Network definition stays in the user's PanFusion checkout; no training imports.
Reference: https://github.com/chengzhag/PanFusion/blob/main/models/faed/FAED.py
"""
import importlib.util
import json
import math
from pathlib import Path
from evaluation.common import FeatureStats, fingerprint_file, frechet, open_rgb, progress


def resize_image(image, height, label, protocol):
    """PanFusion PanoDataset: process_equi uses AREA; pano_pred uses LINEAR.

    Both paths keep RGB. This adapts resizing of already prepared ERP images;
    it does not claim to reproduce an unknown DiT360 preprocessing pipeline.
    """
    import numpy as np
    from PIL import Image
    if image.width != 2 * image.height:
        raise ValueError('FAED requires a 2:1 ERP')
    size = (height * 2, height)
    if protocol == 'pil-bicubic':
        return np.array(image.resize(size, Image.Resampling.BICUBIC))
    if protocol != 'panfusion' or label not in ('real', 'generated'):
        raise ValueError('Unknown FAED preprocessing or set')
    import cv2
    interpolation = cv2.INTER_AREA if label == 'real' else cv2.INTER_LINEAR
    return cv2.resize(np.array(image), size, interpolation=interpolation)


def extract_distributions(encoder, rows, references, args, protocol):
    import torch
    distributions = []
    with torch.inference_mode():
        for label, paths in [('real', references), ('generated', [r['image'] for r in rows])]:
            stats = FeatureStats()
            for start in range(0, len(paths), args.batch_size):
                batch = paths[start:start + args.batch_size]
                images = [torch.from_numpy(resize_image(open_rgb(path), args.faed_height,
                                                       label, protocol)).permute(2, 0, 1)
                          for path in batch]
                x = torch.stack(images).to(args.device).float() / 127.5 - 1
                feature = encoder(x).mean(dim=3)
                latitude = torch.linspace(math.pi / 2, -math.pi / 2, feature.shape[-1], device=args.device)
                feature = (feature * latitude.cos()[None, None, :]).flatten(1)
                stats.update(feature.cpu().numpy())
                progress(f'faed/{protocol}/{label}', start + len(batch), len(paths))
            distributions.append(stats)
    return distributions


def compute(name, rows, references, args):
    import torch
    source = Path(args.panfusion_root) / 'models/faed/modules.py'
    spec = importlib.util.spec_from_file_location('_panfusion_faed_network', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = module.Encoder()
    checkpoint = torch.load(args.faed_weights, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.removeprefix('net.encoder.'): v for k, v in state.items()
             if k.startswith('net.encoder.')}
    if not state:
        raise ValueError('Expected PanFusion faed.ckpt with net.encoder.* state keys')
    encoder.load_state_dict(state, strict=True)
    encoder.eval().to(args.device)
    protocol = getattr(args, 'faed_preprocess', 'pil-bicubic')
    distributions = extract_distributions(encoder, rows, references, args, protocol)
    result = {'value': frechet(*distributions),
            'n_real': distributions[0].n, 'n_generated': distributions[1].n,
            'details': {'backend': 'PanFusion encoder; longitude mean + cosine latitude weighting',
                        'network_sha256': fingerprint_file(source),
                        'weights_sha256': fingerprint_file(args.faed_weights),
                        'height': args.faed_height, 'preprocess': protocol,
                        'resize': 'PIL bicubic' if protocol == 'pil-bicubic' else 'OpenCV real=INTER_AREA; generated=INTER_LINEAR',
                        'distance_backend': 'pytorch-fid scipy sqrtm; PanFusion uses torchmetrics _compute_fid',
                        'protocol_status': 'PanFusion RGB adapter; exact DiT360 encoder/weights/height unverified',
                        'feature_dim': len(distributions[0].mean)}}
    if getattr(args, 'faed_compare_preprocessing', False):
        other = 'panfusion' if protocol == 'pil-bicubic' else 'pil-bicubic'
        other_stats = extract_distributions(encoder, rows, references, args, other)
        values = {protocol: result['value'], other: frechet(*other_stats)}
        diagnostics = {'values': values,
                       'panfusion_minus_pil_bicubic': values['panfusion'] - values['pil-bicubic'],
                       'controlled_variables': result['details'],
                       'note': 'Same images, encoder, weights, height, normalization and distance backend; only resizing changes. Neither value is verified as the DiT360 protocol.'}
        path = Path(args.output) / 'faed_diagnostics.json'
        path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding='utf-8')
        result['details']['preprocessing_comparison'] = diagnostics['values']
        result['details']['diagnostics_file'] = path.name
    return result
