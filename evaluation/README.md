# DiT360 独立评估

只读取生成图片，不加载 DiT360，不训练，不修改原版生成代码。
11 项指标均有计算接口；缺参考集、配置、网络源码或权重时，明确标记 skipped，不填假数值。
这是基于公开实现的独立评估协议，尚未确认与 DiT360 论文表格的全部设置相同。

## 2026-09-09：完整 1092 张的区域 FID 补充与 FAED / IS 诊断

使用已经生成的 `outputs/mp3d_stitched1092_perid_g3_s28`，不重新生成、不重装环境。
从 DiT360 目录，按顺序手动执行。脚本使用现有 `.venv/bin/python`；需要时通过
`PYTHON_BIN` 指向同一套已跑通的 Python。每项输出单独保存，已有目录会拒绝覆盖。

```bash
srun --unbuffered -p debug --nodelist=GPU3 --gres=gpu:1 \
  --qos=normal --time=04:00:00 --mem=64G \
  bash evaluation/run_supplement.sh regions

srun --unbuffered -p debug --nodelist=GPU3 --gres=gpu:1 \
  --qos=normal --time=02:00:00 --mem=64G \
  bash evaluation/run_supplement.sh faed

srun --unbuffered -p debug --nodelist=GPU3 --gres=gpu:1 \
  --qos=normal --time=01:00:00 --mem=64G \
  bash evaluation/run_supplement.sh is

# 仅汇总报告，无模型推理；保留旧八项数值，加上新三项。
bash evaluation/run_supplement.sh summary
```

默认输出 `outputs/mp3d_stitched1092_supplement_v1/`：

| 子目录 | 结果与目的 |
|---|---|
| regions | summary.json/csv：FIDclip、FIDpole、FIDequ |
| faed | summary.json 主值沿用旧 Bicubic；faed_diagnostics.json 对照 PanFusion 缩放 |
| is | summary.json 保留 ResNet18 / splits=1；is_diagnostics.json、两份概率 npz 用于诊断 |
| comparison | comparison.csv/json：原八项＋区域三项与论文的数值对照；不选择最接近论文的诊断结果 |

重跑某阶段时可在该次命令前指定 `RUN_ROOT=outputs/mp3d_stitched1092_supplement_v2`。
summary 阶段只依赖基线和 regions 完成；它核对两份报告的输入清单哈希相同，
不会合并不同样本集合，也不会用诊断分数覆盖原八项。

### 参数依据与公开信息的边界

本次核对 [DiT360 附录 C](https://arxiv.org/html/2510.11712v1#A3)
及官方仓库 `3779fe7965473f6824994c663a0ae7a76bc7aafa`，
[SMGD 官方仓库](https://github.com/chronos123/SMGD)
`6a958558521216911e4e1c99ed2c306543cf1dc5` 的 README / 文件清单。
未找到能确定 DiT360 全部评估参数的官方评估脚本。

- FIDclip：论文要求排除模糊极区，但没有确认具体裁剪量。
  本脚本显式采用每端 0.125，保留中央 75% 高度（1024 高图保留第 128 到 895 行），
  属于本次独立实验选择，不是论文默认值；也不是 CLIP 特征版 FID。
- FIDpole / FIDequ：采用 [SMGD 论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Spherical_Manifold_Guided_Diffusion_Model_for_Panoramic_Image_Generation_CVPR_2025_paper.pdf)
  的 cubemap 极区／赤道分组定义，分别汇集上下 2 面、侧面 4 面的特征计算一个 FID，
  不是分别算每个面的 FID 再取平均。本次选择 py360convert 双线性投影到 512×512；
  这个尺寸、投影实现及 pytorch-fid 后端尚未确认等同于 DiT360。
  1092 张对应每组真实／生成各 2184 个极区面、4368 个赤道面，报告同时保留全景数量。
- 所有 FID 使用同一 pytorch-fid 2048 维 Inception 网络；不改变原 FID。

### FAED 差异如何定位

DiT360 引用 [BIPS](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760331.pdf)
的 FAED 思路。FAED 数值依赖训练出的编码器；引用指标名称不等于确认某个 checkpoint。
当前采用 [PanFusion RGB 编码器与纬度加权](https://github.com/chengzhag/PanFusion/blob/main/models/faed/FAED.py)。

已确认 [PanFusion PanoDataset](https://github.com/chengzhag/PanFusion/blob/main/dataset/PanoDataset.py)
对参考图使用 OpenCV INTER_AREA，对生成图使用默认 INTER_LINEAR；旧实现两侧均为 PIL Bicubic。
`--faed-compare-preprocessing` 在相同图片、编码器、权重、512 高、归一化和距离后端下只改变缩放。
对照针对已经准备好的 ERP，不重建原始参考图，不代表完整复刻 PanFusion 所有数据处理。

读取 `faed_diagnostics.json` 的 `values` 和 `panfusion_minus_pil_bicubic`：
变化小，说明该缩放差异不足以解释 5.96 与 2.91 的落差；变化大，说明预处理敏感，
但仍不能证明 PanFusion 版本就是 DiT360 的计算方式。其余待核对：编码器权重、输入高度、
参考集和准备过程。当前 SciPy sqrtm 与 PanFusion 的 TorchMetrics 距离实现也不完全相同。
PanFusion 中特征维数为输入高度的 4 倍；改变高度同时改变编码器输入与特征维数，
因此即使 5.96 接近 2.91 的两倍，也不能直接除以 2 或据此认定论文使用了 256 高。
不能通过修改高度、权重或后端来挑选最接近论文的数字。

### IS 差异如何定位

DiT360 指定 Places365 ResNet，但未确认层数、权重、输入变换和 splits。
当前 ResNet18 的 resize256×256 / center224 / ImageNet normalization 来自
[Places365 官方示例](https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py)。
PanFusion EvalPanoGen 使用标准 InceptionScore，因此不能直接用其 IS 替代 DiT360 的 Places365 IS。

`--is-diagnostics` 不改变主值（splits=1）：

- 保存生成图／参考图的完整分类概率、ID、路径，后续分析不必重复跑分类器。
- 给出全局类别熵 H(Y)、平均单图类别熵 H(Y|X)，核对 IS = exp(H(Y)-H(Y|X))。
- 同一概率矩阵对照原顺序／固定种子 0 打乱的 1、5、10 分组；不丢余数样本。
- 给出按 Matterport 建筑 ID 分组的诊断和参考图的同模型统计。参考 IS 不是新的论文指标。

如果原顺序分组与打乱分组差异明显，说明场景排序会影响分组 IS；仍不代表论文使用了其中一种。
若各种分组均不能解释落差，应优先核对分类器、权重及预处理，而不是修改图片或凭数值选协议。
splits=1 的 std=0 只表示单组，不是多次生成的置信区间。

以上诊断不自动判定“差异全部来自实现”，不代表生成质量超过或低于论文。

## 文件结构

| 文件 | 用途 |
|---|---|
| evaluate.py | 统一命令行入口、配置检查、实时进度、汇总与错误报告 |
| common.py | 读图、输入清单、统计计算 |
| metrics/fid.py | FID、FIDclip、FIDpole、FIDequ，复用同一特征网络 |
| metrics/faed.py | PanFusion 自编码器特征与 FAED |
| metrics/places_is.py | Places365 ResNet 分类概率与 IS |
| metrics/clip_score.py | 提示词与图片的 CS |
| metrics/no_reference.py | Q-Align quality/aesthetic、BRISQUE、NIQE |
| tests/ | 不加载预训练模型的 CPU 统计与适配器测试 |

同族指标共用代码，避免为了“每个指标一个文件”重复读图和统计逻辑。

## 真实参考图集是什么

FID 的四种形式及 FAED 比较两组图片的特征分布：生成图集 vs 真实场景参考图集。
不要求一一像素对应，不是 PSNR/SSIM 的逐图比较。验证集的场景分布、划分和图像处理必须合适。
其余六项不需要真实参考图；CS 需要每张图的实际生成提示词。

2026-09-07 已核查本机缓存的 dataset_info.json：

```
/data-nfs/gpu1-2/u13529658780/.cache/huggingface/datasets/Insta360-Research___matterport3_d_polished/
```

只有 train，共 10,359 条，字段 image/caption；没有 validation/test。
这是作者公开的 polished 训练数据，不能称为论文真实验证集。极区经过修整，也不能直接冒充原始实拍参考。
从作者训练数据重新抽出一部分，不能消除作者预训练模型已经见过它们的问题。
正式评测应准备 Matterport3D 相应验证划分及描述；可参照 PanFusion/MVDiffusion 数据准备，
但仍需核对 DiT360 的实际验证清单。室外自建提示词不可直接与室内参考集混比。
本工具不自动下载、划分或导出该训练缓存，也不把 HF Arrow 缓存路径当图片目录。

## 安装（用户执行）

建议新建环境，避免 pyiqa 的 transformers 依赖改动已跑通的 DiT360 环境。
下面选择与现有 PyTorch 主版本一致的独立环境；未在本次安装或验证完整依赖解析。

```bash
cd /data-nfs/gpu1-2/u13529658780/DIT360/DiT360
conda create -n dit360-eval python=3.11 -y
conda activate dit360-eval
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r evaluation/requirements.txt
python -m pip check
```

pyiqa 固定 0.1.14.1，使用其依赖的 transformers 4.37.2；不用最新主分支的 Q-ReAlign 替代 Q-Align。
首次实际计算时，FID/CLIP/pyiqa 可能下载对应评估模型。FAED 和 Places365 权重需提前手动准备。
Q-Align 是较大的模型，建议在 GPU2/GPU3 单卡串行评估，不与生成模型同时占同一张卡。

## 输入和输出

从 DiT360 根目录运行 `python -m evaluation.evaluate`。
`--generated-dir` 递归读取图片，并从同名 `.json` 中取 prompt/id/seed 等元数据，兼容当前 inference_copy.py。
目录内应只包含本轮待评估图片，不混入缩略图、拼图、以前的 seed 或真实参考图。
运行前先等该轮生成全部结束，避免只评到部分图片或正在写入的图片。

也可以用 `--generated-manifest input.jsonl` 精确选样本。每行：

```json
{"image":"../outputs/run/implicit_01_seed0.png","prompt":"This is a panorama. A runner on a sports field in the rain."}
```

路径相对 JSONL 所在目录解析，也接受绝对路径。`--reference-manifest` 同样格式，但无需 prompt。
原提示词文件只有 prompt 没有 image 路径，不能直接当作该评估 manifest。
不自动把检查标准等元数据拼入 CS 文本。

输出目录必须是新的目录，防止旧报告被覆盖：

- summary.csv：每个指标的数值、方向、状态、样本数及未完成原因。
- summary.json：额外保存参数、依赖版本、源码哈希、各指标协议详情和时间。
- per_image.csv：CS/QA/BRISQUE/NIQE 的逐图分数；分布指标及 IS 不伪造逐图分数。
- inputs.jsonl：冻结本轮有序图片路径、文件大小/修改时间与生成元数据。不是图片内容哈希。
- `<metric>_error.txt`：仅失败时生成完整错误信息。

逐图指标的 value 是均值，std 是总体标准差，不是置信区间；IS 的 std 是分组得分的总体标准差。
每完成一项立即更新汇总，某项失败继续计算其他项。中断保留已完成指标；重新计算使用新输出目录。
退出码：0=选中指标全部成功；2=有 skipped/error；130=手动中断。
dry-run 的 0 仅表示配置检查完成，应查看 ready/skipped；不检查 GPU 模型执行是否成功。

## 当前先运行无需参考集的指标

检查输入/配置，不用 GPU，不加载或下载评估模型：

```bash
python -m evaluation.evaluate \
  --generated-dir outputs/dit360_world_knowledge_panorama_prefix_v2 \
  --output outputs/eval_prefix_v2_check \
  --metrics all \
  --clip-model openai/clip-vit-base-patch16 \
  --dry-run
```

实际先算 BRISQUE、NIQE、CS：

```bash
srun --unbuffered -p debug --nodelist=GPU3 --gres=gpu:1 \
  --qos=normal --time=02:00:00 \
  python -u -m evaluation.evaluate \
  --generated-dir outputs/dit360_world_knowledge_panorama_prefix_v2 \
  --output outputs/eval_prefix_v2_basic \
  --metrics brisque niqe cs \
  --clip-model openai/clip-vit-base-patch16
```

`openai/clip-vit-base-patch16` 是这里显式选择的基线评测配置，参考 PanFusion；并未确认是 DiT360 使用的型号。
CS 使用 `max(100*cosine, 0)`，完整 ERP 交给 CLIP 默认预处理；不自动变成多视角平均。
文本超过 CLIP 长度时遵循后端行为，注意控制长度并查看警告。

算上两项 Q-Align 时，把 metrics 改为：

```text
--metrics brisque niqe cs qaquality qaaesthetic
```

BRISQUE/NIQE 采用 pyiqa 同名实现，不混用 `_matlab` 变体。两项 QA 都使用 q-future/one-align。
每项顺序加载并释放，默认 batch-size=1；模型初次下载时等待与评分进度不同。

## 有真实验证集后计算分布指标

### 单独指定 Q-Align 本地模型（2026-09-08）

`--qalign-model-path` 只影响 Q-Align 的权重、tokenizer 和图像处理器，
不修改 HF_HOME、TRANSFORMERS_CACHE 或其他指标的加载路径。
传入 `snapshots/<revision>` 目录，而不是 `blobs` 或缓存仓库顶层。
本地模式使用 `local_files_only=True` 与 `use_safetensors=False`，只读已有 .bin 分片，缺文件直接报错。
该模式适配当前安装的 pyiqa 0.1.15 内置 q_align 实现；旧版 0.1.14.1 不支持该新模式，
会明确报错，不自动升级环境或改用远程模型代码。其余原有指标入口仍可使用。

```bash
srun --unbuffered -p debug --nodelist=GPU3 --gres=gpu:1 \
  --qos=normal --time=04:00:00 \
  python -u -m evaluation.evaluate \
  --generated-dir outputs/dit360_world_knowledge_panorama_author \
  --output outputs/panorama_author_qa_local \
  --metrics qaquality qaaesthetic \
  --qalign-model-path /data-nfs/gpu1-2/u13529658780/.cache/huggingface/hub/models--q-future--one-align/snapshots/dcc603b95aa0ebd82afa696d4a1e20d11fc80ddb
```

该选项不需要设置全局离线变量，也不会改变已启动进程；下次启动才生效。
保持 FP16 和原 Q-Align 评分方式，不自动量化。仅做过本地文件与加载接口测试，未运行真实权重的 GPU 推理。

基本 FID 命令示例（路径需要替换为真实存在的验证图片目录）：

```bash
srun --unbuffered -p debug --nodelist=GPU3 --gres=gpu:1 \
  --qos=normal --time=04:00:00 \
  python -u -m evaluation.evaluate \
  --generated-dir /path/to/generated_validation \
  --reference-dir /path/to/real_validation \
  --reference-label "Matterport3D validation; describe exact split/version here" \
  --output outputs/eval_validation_fid \
  --metrics fid
```

小样本也允许计算，但估计可能不稳定。脚本对少于1000张给出提醒，
这不是论文规定，也不代表1000张就足够稳定；不会自动增删样本或阻止计算。
输出保留实际样本量，不把20/40张诊断结果用于论文表格的直接横比。

各项额外参数：

| 指标 | 必需参数 | 当前实现和未确认事项 |
|---|---|---|
| fid | 参考集及 reference-label | pytorch-fid 2048 维，后端缩放到299×299 |
| fidclip | `--clip-crop-fraction` | 每端去除 floor(H×fraction) 行；作者比例未确认，无默认值 |
| fidpole/fidequ | `--face-size` | py360convert 双线性，U/D 两面或 F/R/B/L 四面分别汇入一个集合；作者面尺寸未确认 |
| faed | `--panfusion-root`、`--faed-weights`、`--faed-height` | 高度为32倍数；PIL bicubic；PanFusion 编码器及纬度加权；DiT360 权重/缩放设置未确认 |
| is | `--places-weights`、`--places-arch`、`--is-splits` | resnet18/resnet50，365类；Places 官方示例预处理；作者具体型号和分组未确认 |

例如 `--face-size 256`、`--faed-height 512` 可以作为你们自行约定的固定协议，但不能标为论文原值。
FIDclip 的 clip 是裁掉极区的含义，不是 CLIP 特征；IS 不用默认 Inception-v3 代替。
FIDpole/equ 计数是投影面数，不是原全景数；summary 顶层另记录原全景数量。

FAED：从 PanFusion 官方仓库取网络定义与 README 链接的 faed.ckpt，无需训练。

```bash
git clone https://github.com/chengzhag/PanFusion /path/to/PanFusion
```

本适配器只读取该仓库的 `models/faed/modules.py`，不导入它的训练/日志系统。
`--faed-weights /path/to/faed.ckpt` 接受 PanFusion Lightning checkpoint 的 `net.encoder.*` 权重。
权重和网络源码哈希写入报告。Places365 从官方仓库下载对应 ResNet 的 `.pth.tar`，
架构和权重必须匹配。两者均使用 `torch.load(weights_only=True)`，不自动放宽加载方式。

全部材料准备好后，使用同一个入口 `--metrics all` 并提供上述参数，即可汇总11项。
参考集路径、提示词、模型版本、数量、分辨率、预处理与随机种子列表必须在方法间固定。
不同 CLIP、FAED、裁剪或投影配置的数值不能混入同一对比表。

## 验证与来源

### 本地 CLIP 与全部指标入口（2026-09-08）

新增 `--clip-model-path /path/to/snapshot`：只让 CS 的 CLIP 权重和 processor 使用本地文件，
固定 `.bin` 与 slow processor，不改代理或 HF 缓存环境变量。路径必须含完整配置、词表、merges 和 pytorch_model.bin。
可同时保留 `--clip-model openai/clip-vit-base-patch16` 作为报告里的模型名称。
当前缓存目录：

```
/data-nfs/gpu1-2/u13529658780/.cache/huggingface/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3
```

`--metrics all` 才选择全部11项；输出目录名称不会影响选择。
材料未齐时，全选仍会出现 skipped，不能称为完成全部评测。
当前环境若沿用 pyiqa 0.1.15 本地 Q-Align，请勿重新安装旧 requirements 导致降级。
仅补当前缺少的两个后端包：

```bash
python -m pip install 'pytorch-fid==0.3.0' 'py360convert>=1,<2'
python -m pip check
```

完整命令模板如下。先给变量填写已准备的路径和已确定的评测协议；
不提供虚构的论文裁剪比例或参考集路径。GEN_DIR应是对应验证描述的生成集。

```bash
# 必须提前设置：GEN_DIR REAL_DIR REFERENCE_LABEL CLIP_PATH QALIGN_PATH
# PANFUSION_ROOT FAED_WEIGHTS FAED_HEIGHT PLACES_WEIGHTS PLACES_ARCH IS_SPLITS
# FIDCLIP_CROP FACE_SIZE EVAL_OUTPUT
: "${GEN_DIR:?}" "${REAL_DIR:?}" "${REFERENCE_LABEL:?}" "${CLIP_PATH:?}" "${QALIGN_PATH:?}"
: "${PANFUSION_ROOT:?}" "${FAED_WEIGHTS:?}" "${FAED_HEIGHT:?}" "${PLACES_WEIGHTS:?}"
: "${PLACES_ARCH:?}" "${IS_SPLITS:?}" "${FIDCLIP_CROP:?}" "${FACE_SIZE:?}" "${EVAL_OUTPUT:?}"
srun --unbuffered -p debug --nodelist=GPU3 --gres=gpu:1 \
  --qos=normal --time=04:00:00 \
  python -u -m evaluation.evaluate \
  --generated-dir "$GEN_DIR" --reference-dir "$REAL_DIR" \
  --reference-label "$REFERENCE_LABEL" --output "$EVAL_OUTPUT" --metrics all \
  --clip-model openai/clip-vit-base-patch16 --clip-model-path "$CLIP_PATH" \
  --qalign-model-path "$QALIGN_PATH" \
  --clip-crop-fraction "$FIDCLIP_CROP" --face-size "$FACE_SIZE" \
  --panfusion-root "$PANFUSION_ROOT" --faed-weights "$FAED_WEIGHTS" --faed-height "$FAED_HEIGHT" \
  --places-weights "$PLACES_WEIGHTS" --places-arch "$PLACES_ARCH" --is-splits "$IS_SPLITS"
```

先追加 `--dry-run` 检查配置（无需srun），确认全部ready后再实际运行。
实际耗时取决于验证集规模，上述4小时是作业上限示例。

本次只进行语法、CPU 统计/输入/汇总测试和 dry-run，不执行真实图片指标计算、GPU 作业或模型下载。
完整依赖安装、评估权重与 GPU 端到端运行仍需在服务器执行后确认。

```bash
python -m unittest discover -s evaluation/tests -v
```

- DiT360 指标定义：https://arxiv.org/html/2510.11712v1#A3
- 公开训练集：https://huggingface.co/datasets/Insta360-Research/Matterport3D_polished
- PanFusion 验证数据准备及 FAED 权重：https://github.com/chengzhag/PanFusion
- FAED：https://github.com/chengzhag/PanFusion/blob/main/models/faed/FAED.py
- FID：https://github.com/mseitzer/pytorch-fid
- 球面分组 FID：https://github.com/chronos123/SMGD
- 投影：https://github.com/sunset1995/py360convert
- IS 分类器及预处理：https://github.com/CSAILVision/places365
- CS：https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/multimodal/clip_score.py
- Q-Align：https://github.com/q-future/q-align
- IQA：https://github.com/chaofengc/IQA-PyTorch/tree/v0.1.14.1

这些通用指标不能单独证明因果推理能力；仍应人工检查提示条件、物理依赖和全景各朝向的一致性。
