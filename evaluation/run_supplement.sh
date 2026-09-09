#!/usr/bin/env bash
# Run under srun. This script does not allocate GPUs or regenerate images.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-outputs/mp3d_stitched1092_supplement_v1}"
common=(--prepared ../benchmark_assets/mp3d_stitched1092
        --generated outputs/mp3d_stitched1092_perid_g3_s28
        --device cuda:0 --batch-size 1)

case "${1:-}" in
  regions)
    # Explicit independent choices, NOT published DiT360 defaults:
    # remove 12.5% from EACH pole, retain central 75%; cube faces 512x512.
    "$PYTHON_BIN" -u -m evaluation.benchmark "${common[@]}" \
      --output "$RUN_ROOT/regions" --metrics fidclip fidpole fidequ \
      --clip-crop-fraction 0.125 --face-size 512
    ;;
  faed)
    "$PYTHON_BIN" -u -m evaluation.benchmark "${common[@]}" \
      --output "$RUN_ROOT/faed" --metrics faed \
      --panfusion-root ../benchmark_assets/PanFusion \
      --faed-weights ../benchmark_assets/weights/faed.ckpt --faed-height 512 \
      --faed-preprocess pil-bicubic --faed-compare-preprocessing
    ;;
  is)
    "$PYTHON_BIN" -u -m evaluation.benchmark "${common[@]}" \
      --output "$RUN_ROOT/is" --metrics is \
      --places-weights ../benchmark_assets/weights/resnet18_places365.pth.tar \
      --places-arch resnet18 --is-splits 1 --is-diagnostics
    ;;
  summary)
    "$PYTHON_BIN" -u -m evaluation.compare_reports \
      --baseline outputs/mp3d_stitched1092_perid_eval8_v1/summary.json \
      --regions "$RUN_ROOT/regions/summary.json" --output "$RUN_ROOT/comparison"
    ;;
  *)
    echo 'Usage: bash evaluation/run_supplement.sh {regions|faed|is|summary}' >&2
    exit 2
    ;;
esac
