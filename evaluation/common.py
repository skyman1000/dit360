"""Input validation, streaming statistics and result helpers."""
import hashlib
import json
import math
from pathlib import Path

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def collect_images(directory):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f'Image directory does not exist: {root}')
    paths = sorted(p.resolve() for p in root.rglob('*')
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f'No images found: {root}')
    return paths


def load_samples(directory=None, manifest=None):
    """Manifest paths are relative to the manifest, never to shell cwd."""
    if manifest:
        manifest = Path(manifest).resolve()
        rows = []
        for line in manifest.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(row['image'])
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.is_file():
                raise ValueError(f'Missing image: {path}')
            rows.append({**row, 'image': str(path.resolve())})
    else:
        rows = []
        for path in collect_images(directory):
            metadata = path.with_suffix('.json')
            row = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {}
            rows.append({**row, 'image': str(path)})
    if not rows:
        raise ValueError('Empty generated manifest')
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate generated image paths')
    return rows


def fingerprint_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def open_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def image_tensor(path):
    import numpy as np
    import torch
    return torch.from_numpy(np.array(open_rgb(path))).permute(2, 0, 1)


def progress(label, index, total):
    print(f'[{label}] {index}/{total}', flush=True)


def per_image_result(values, details):
    import numpy as np
    scores = np.array(list(values.values()), dtype=np.float64)
    if not scores.size or not np.isfinite(scores).all():
        raise ValueError('Empty or non-finite metric scores')
    return {'value': float(scores.mean()), 'std': float(scores.std()),
            'n': len(scores), 'per_image': values, 'details': details}


class FeatureStats:
    """Merge centered batch moments in float64 without retaining all features."""
    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, features):
        import numpy as np
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError('Invalid feature batch')
        mean = x.mean(0)
        centered = x - mean
        m2 = centered.T @ centered
        if self.n == 0:
            self.n, self.mean, self.m2 = len(x), mean, m2
            return
        delta = mean - self.mean
        total = self.n + len(x)
        self.m2 += m2 + np.outer(delta, delta) * (self.n * len(x) / total)
        self.mean += delta * (len(x) / total)
        self.n = total

    def covariance(self):
        if self.n < 2:
            raise ValueError('Distribution metrics need at least two samples per set')
        return self.m2 / (self.n - 1)


def frechet(real, generated):
    from pytorch_fid.fid_score import calculate_frechet_distance
    value = float(calculate_frechet_distance(
        real.mean, real.covariance(), generated.mean, generated.covariance()))
    if not math.isfinite(value) or value < -1e-5:
        raise ValueError(f'Invalid Frechet distance: {value}')
    return max(0.0, value)


def crop_bounds(height, fraction):
    if not 0 < fraction < 0.5:
        raise ValueError('Polar crop fraction must be between 0 and 0.5')
    cut = int(height * fraction)
    if cut < 1 or height - 2 * cut < 1:
        raise ValueError('Crop removes no pixels or the entire image')
    return cut, height - cut


def inception_score(probabilities, splits):
    import numpy as np
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Invalid class probabilities')
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Class probabilities must sum to one')
    if not 1 <= splits <= len(p):
        raise ValueError('IS splits must be between 1 and image count')
    scores = []
    for part in np.array_split(p, splits):
        marginal = part.mean(0, keepdims=True)
        kl = (part * (np.log(np.maximum(part, 1e-300)) -
                      np.log(np.maximum(marginal, 1e-300)))).sum(1)
        scores.append(float(np.exp(kl.mean())))
    return float(np.mean(scores)), float(np.std(scores)), scores
