"""IS with an explicitly selected Places365 ResNet and checkpoint."""
import json
from pathlib import Path
import re
from evaluation.common import fingerprint_file, inception_score, open_rgb, progress


def probability_diagnostics(probabilities, rows):
    """Sensitivity diagnostics, never an automatic selection of the best IS."""
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    global_is = inception_score(p, 1)[0]  # also validates the probabilities
    marginal = p.mean(0)
    conditional_entropy = float(-(p * np.log(np.maximum(p, 1e-300))).sum(1).mean())
    marginal_entropy = float(-(marginal * np.log(np.maximum(marginal, 1e-300))).sum())
    split_results = {}
    permutation = np.random.default_rng(0).permutation(len(p))
    for splits in (1, 5, 10):
        if splits > len(p):
            continue
        for label, q in [('manifest_order', p), ('shuffled_seed0', p[permutation])]:
            mean, std, scores = inception_score(q, splits)
            split_results[f'{label}/splits{splits}'] = {'value': mean, 'std': std, 'scores': scores}
    buildings = {}
    for i, row in enumerate(rows):
        sample_id = str(row.get('id', ''))
        if re.match(r'^[A-Za-z0-9]{11}_[0-9a-f]{32}$', sample_id):
            buildings.setdefault(sample_id.split('_')[0], []).append(i)
    per_building = {building: {'n': len(indices), 'value': inception_score(p[indices], 1)[0]}
                    for building, indices in buildings.items()}
    return {
        'n': len(p), 'classes': p.shape[1], 'global_is': global_is,
        'marginal_entropy_nats': marginal_entropy,
        'mean_conditional_entropy_nats': conditional_entropy,
        'exp_entropy_difference': float(np.exp(marginal_entropy - conditional_entropy)),
        'mean_top1_probability': float(p.max(1).mean()),
        'unique_top1_classes': int(len(np.unique(p.argmax(1)))),
        'top10_marginal_classes': [{'class_index': int(i), 'probability': float(marginal[i])}
                                  for i in np.argsort(-marginal)[:10]],
        'split_sensitivity': split_results, 'per_building': per_building,
        'note': 'Diagnostic variants, not paper scores. Split std is not multi-seed uncertainty. Building groups recognized only from Matterport view IDs; no images dropped from global IS.'}


def collect_probabilities(model, transform, rows, args, label):
    import numpy as np
    import torch
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            x = torch.stack([transform(open_rgb(r['image'])) for r in batch]).to(args.device)
            probabilities.append(model(x).softmax(1).cpu().numpy())
            progress(label, start + len(batch), len(rows))
    return np.concatenate(probabilities)


def compute(name, rows, references, args):
    import numpy as np
    import torch
    from torchvision import models, transforms
    model = getattr(models, args.places_arch)(weights=None, num_classes=365)
    checkpoint = torch.load(args.places_weights, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)
    # Matches the public Places365 run_placesCNN_basic.py example.
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    p = collect_probabilities(model, transform, rows, args, 'is/generated')
    # Stable input order, no hidden random shuffle; all images are included.
    mean, std, scores = inception_score(p, args.is_splits)
    result = {'value': mean, 'std': std, 'n': len(rows), 'split_scores': scores,
            'details': {'backend': 'torchvision Places365 ResNet + exp(mean KL)',
                        'architecture': args.places_arch,
                        'weights_sha256': fingerprint_file(args.places_weights),
                        'splits': args.is_splits, 'split_order': 'manifest/deterministic path order',
                        'preprocess': 'PIL RGB resize256x256 bilinear, center224, ImageNet normalization',
                        'protocol_status': 'Places365 backbone family follows DiT360; architecture/checkpoint/preprocess/splits unverified'}}
    if getattr(args, 'is_diagnostics', False):
        output = Path(args.output)
        diagnostics = {'protocol': dict(result['details']), 'generated': probability_diagnostics(p, rows)}
        def save_probabilities(label, probabilities, items):
            path = output / f'is_probabilities_{label}.npz'
            np.savez_compressed(path, probabilities=probabilities,
                                ids=np.array([str(r.get('id', '')) for r in items]),
                                images=np.array([str(r['image']) for r in items]))
            return {'file': path.name, 'sha256': fingerprint_file(path)}
        diagnostics['generated']['artifact'] = save_probabilities('generated', p, rows)
        # Reference IS is a distribution diagnostic, not an additional paper metric.
        if references:
            ref_rows = [{'image': str(path), 'id': Path(path).stem} for path in references]
            ref_p = collect_probabilities(model, transform, ref_rows, args, 'is/reference-diagnostic')
            diagnostics['reference'] = probability_diagnostics(ref_p, ref_rows)
            diagnostics['reference']['artifact'] = save_probabilities('reference', ref_p, ref_rows)
        path = output / 'is_diagnostics.json'
        path.write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding='utf-8')
        result['details']['diagnostics_file'] = path.name
    return result
