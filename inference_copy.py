import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from src.pipeline import DiT360Pipeline

model_id = "black-forest-labs/FLUX.1-dev"
lora_id = "Insta360-Research/DiT360-Panorama-Image-Generation"

model_path = snapshot_download(model_id, local_files_only=True)
lora_path = snapshot_download(lora_id, local_files_only=True)

rows = [
    json.loads(line)
    for line in Path(
        "experiments/world_knowledge_v1/implicit_20.jsonl"
    ).read_text().splitlines()
    if line.strip()
]

out = Path("outputs/dit360_world_knowledge_panorama_implicit")
out.mkdir(parents=True, exist_ok=True)

pipe = DiT360Pipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    local_files_only=True,
).to("cuda:0")

pipe.load_lora_weights(
    lora_path,
    weight_name="adapter_model.safetensors",
    local_files_only=True,
)

for i, row in enumerate(rows, 1):
    image_path = out / f"{row['id']}_seed0.png"
    metadata_path = image_path.with_suffix(".json")

    metadata = {
        **row,
        "model": model_id,
        "model_snapshot": model_path,
        "lora": lora_id,
        "lora_snapshot": lora_path,
        "seed": 0,
        "width": 2048,
        "height": 1024,
        "steps": 28,
        "guidance_scale": 3.0,
        "dtype": "float16",
    }

    if image_path.exists() and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous != metadata:
            raise RuntimeError(f"已有结果配置不同，请换输出目录：{image_path}")
        print(f"[{i}/{len(rows)}] 已完成，跳过 {row['id']}", flush=True)
        continue

    print(f"[{i}/{len(rows)}] 正在生成 {row['id']}", flush=True)

    image = pipe(
        row["prompt"],
        width=2048,
        height=1024,
        num_inference_steps=28,
        guidance_scale=3.0,
        generator=torch.Generator(device="cuda:0").manual_seed(0),
    ).images[0]

    image.save(image_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )

print("20 条提示词推理完成。", flush=True)
