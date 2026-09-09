"""pyiqa adapters; full ERP input, one image at a time."""
from evaluation.common import image_tensor, per_image_result, progress


def compute(name, rows, references, args):
    import torch
    import pyiqa
    backend = {'qaquality': 'qalign', 'qaaesthetic': 'qalign'}.get(name, name)
    task = {'qaquality': 'quality', 'qaaesthetic': 'aesthetic'}.get(name)
    local_path = getattr(args, 'qalign_model_path', None) if task else None
    if local_path:
        from evaluation.metrics.qalign_local import create_local_metric
        print(f'[{name}] Local Q-Align snapshot: {local_path} (offline .bin loading)', flush=True)
        metric = create_local_metric(local_path, args.device)
    else:
        metric = pyiqa.create_metric(backend, device=args.device).eval()
    values = {}
    with torch.inference_mode():
        for i, row in enumerate(rows, 1):
            x = image_tensor(row['image']).unsqueeze(0).to(args.device).float() / 255
            score = metric(x, **({'task_': task} if task else {}))
            values[row['image']] = float(score.item())
            progress(name, i, len(rows))
    return per_image_result(values, {'backend': f'pyiqa/{backend}', 'task': task,
                                    'input': 'full ERP RGB float32 [0,1]; backend preprocessing',
                                    'qalign_model': 'q-future/one-align' if task else None,
                                    'qalign_local_snapshot': local_path,
                                    'qalign_local_files_only': bool(local_path)})
