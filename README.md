# SAD-Bridge

**Official inference & feature-extraction code for SAD-Bridge**
(Self-consistency Anomalous-gradient Detector) — a training-free,
gradient-based AI-generated image detector built on top of
`facebook/metaclip-2-worldwide-giant`.

- **Author:** Xinan He
- **Date:** 2026-07-10

SAD-Bridge detects AI-generated images without any classifier training.
It probes MetaCLIP-2's vision encoder with a label-free self-consistency
loss (rotation invariance across four views) and uses the **gradient**
of that loss with respect to the CLS token at an intermediate layer as
the forensic feature. Real and fake images produce gradients with
systematically different geometry, which a simple k-NN classifier against
a small reference set can separate with high accuracy — generalizing
across generators, resolutions, and domains without ever training a
detector head.

## Validated best configuration

| Component        | Value                                              |
|-------------------|-----------------------------------------------------|
| Backbone          | `facebook/metaclip-2-worldwide-giant`                |
| Geometry views    | `rotation` — anchor(0°) + 90° + 180° + 270°          |
| Self-consistency loss | MSE between CLS(anchor) and CLS(rotated views) |
| Gradient aggregation | `diff` = grad[view0, CLS] − mean(grad[view≥1, CLS]) |
| Feature layer     | hidden layer **32** (`layer_32_cls_grad`, dim 1664) |
| Classifier        | k-NN, **k = 25**, cosine similarity                  |
| Reference set     | GenImage / `stable_diffusion_v_1_4` / **train** split, 500 real + 500 fake sampled with seed **1314** |

This configuration was selected as **rank-1** across a joint sweep over
seeds `{1, 42, 256, 512, 1024, 1314, 1729, 8888, 65535}`, layers
`{0,4,...,48}`, and k `{5,...,15,20,25}`, evaluated on a broad portfolio
of held-out AI-image-forensics benchmarks (GenImage, GenImage++,
Community-Forensics, AIGI_Holmes, DRCT, synthbuster, WildRF, CO-SPYBench,
corebench, NTIRE, RealChain, anime_test, and more). Full per-dataset
results for this exact configuration:
`checkpoints/metaclip2_clsgrad_sdv14ref_rot_diff/results/knn_k5-6-7-8-9-10-11-12-13-14-15-20-25_ref500_seeds_1-42-256-512-1024-1314-1729-8888-65535/focus_datasets/portfolio_rank1_seed1314_layer_32_cls_grad_k25_all_targets.csv`

## Files

| File | Purpose |
|------|---------|
| `extract_features.py` | Extracts the 1664-d SAD-Bridge feature for a folder of images. Used both to build a new **reference set** and to pre-extract **query features**. |
| `predict.py` | Runs k-NN(k=25) classification of query images/features against a reference set. This is the detector's inference entry point. |

## Installation

```bash
pip install -r requirements.txt
```

Requires network/cache access to download `facebook/metaclip-2-worldwide-giant`
from the HuggingFace Hub (≈ giant-sized ViT, first run will download weights).

## 1. Feature extraction (`extract_features.py`)

Extracts the SAD-Bridge descriptor for every image under one or more
`--image-dir` paths (recursively scanned; supports `.jpg/.jpeg/.png/.webp/.bmp/.tiff/.tif`).

```bash
python extract_features.py \
    --image-dir /path/to/images \
    --out features.pt \
    --device cuda:0
```

If a folder follows the `0_real/` / `1_fake/` convention (immediate parent
folder name of each image), ground-truth labels are attached automatically —
this is required when building a **reference set** (see below), and optional
for query images (only used to compute accuracy/AUC after prediction).

```bash
python extract_features.py \
    --image-dir /path/to/dataset/0_real \
    --image-dir /path/to/dataset/1_fake \
    --out features.pt
```

### Arguments

| Flag | Default | Description |
|------|---------|--------------|
| `--image-dir` | *(required, repeatable)* | Image file or directory. Repeat the flag to combine multiple sources into one output. |
| `--out` | *(required)* | Output `.pt` path. |
| `--device` | `cuda:0` | Falls back to `cpu` automatically if CUDA unavailable. |
| `--num-workers` | `4` | DataLoader workers for image decoding/preprocessing. |
| `--model-name` | `facebook/metaclip-2-worldwide-giant` | HF model id/path. Do not change unless re-validating a different backbone. |

### Output format

`features.pt` is a `torch.save` dict:

```python
{
    "features":    Tensor[N, 1664]  float16,   # SAD-Bridge descriptor (NOT L2-normalised)
    "labels":      Tensor[N]        int64,     # 0=real, 1=fake, -1=unknown
    "image_paths": List[str],                  # absolute paths, aligned with rows above
    "model_name":  str,
    "feature_key": "layer_32_cls_grad",
    "layer":       32,
}
```

### How it works internally

For each image:
1. Preprocess with MetaCLIP-2's `AutoProcessor` → pixel tensor.
2. Build 4 geometric views: `[0°, 90°, 180°, 270°]` rotations of the same image.
3. Forward all 4 views through `model.vision_model(output_hidden_states=True)`.
4. Compute the self-consistency loss: mean MSE between the CLS token of the
   anchor (0°) view and each rotated view, **at layer 32 only**.
5. Backpropagate; read `hidden_states[32].grad`, shape `(4, seq, 1664)`.
6. Aggregate across views with `diff`: `grad[0, CLS] − mean(grad[1:, CLS])`.

No model weights are updated — this is a fully training-free, per-image,
gradient-probing feature extractor.

## 2. Building / updating the reference (gallery) set

The reference set is just a feature cache with ground-truth labels,
produced by the exact same `extract_features.py` script. The published
best configuration uses **GenImage `stable_diffusion_v_1_4` train split**
(`0_real` / `1_fake` sub-folders), sub-sampled to **500 real + 500 fake**
images with a fixed seed **1314**.

```bash
python extract_features.py \
    --image-dir /path/to/GenImage/stable_diffusion_v_1_4/train/0_real \
    --image-dir /path/to/GenImage/stable_diffusion_v_1_4/train/1_fake \
    --out reference.pt \
    --device cuda:0
```

`predict.py` performs the 500-per-class / seed-1314 sub-sampling and
L2-normalisation **at load time** — you do not need to pre-sample the
reference set yourself, just point `--ref-cache` at the full-size
extracted feature cache (any number of real/fake images ≥ 500 per class
is fine; extra samples are simply not drawn).

To use a **different** reference domain/generator, extract features from
that labelled corpus the same way and pass `--ref-cache your_reference.pt`
to `predict.py`. Reference sub-sampling parameters (`k=25`, `seed=1314`,
`n=500/class`) are the validated defaults baked into `predict.py`; edit
the `KNN_K` / `REF_SEED` / `REF_N` constants at the top of that file if
you need to re-tune them for a new domain.

`predict.py`'s built-in default `--ref-cache` points at
`../checkpoints/metaclip2_clsgrad_sdv14ref_rot_diff/features/GenImage__stable_diffusion_v_1_4__train.pt`
(relative to the script), which is where the pre-built reference cache from
the original training run lives in the source [Continual_Learning](https://github.com/XinanHe-UB/Continual_Learning)
repository. This path is **not included** in this standalone repository
(feature caches are large and not version-controlled here) — you must
either:

- build your own reference cache as described above and pass
  `--ref-cache reference.pt` explicitly, or
- copy the pre-built cache file from the source repo into this directory
  layout, or
- edit `_DEFAULT_REF_CACHE` at the top of `predict.py` to point at wherever
  you keep it.

## 3. Inference (`predict.py`)

### Option A — pre-extracted query features (recommended for large batches)

```bash
python extract_features.py --image-dir /path/to/query_images --out query.pt
python predict.py --features query.pt --out predictions.csv
```

### Option B — extract + predict in one call

```bash
python predict.py \
    --image-dir /path/to/query_images \
    --out predictions.csv \
    --device cuda:0
```

### Option C — custom reference set

```bash
python predict.py \
    --features query.pt \
    --ref-cache /path/to/your_reference.pt \
    --out predictions.csv
```

### Arguments

| Flag | Default | Description |
|------|---------|--------------|
| `--features` | *(mutually exclusive with `--image-dir`)* | Pre-extracted `.pt` from `extract_features.py`. |
| `--image-dir` | *(mutually exclusive with `--features`, repeatable)* | Extract query features on the fly. |
| `--ref-cache` | `../checkpoints/metaclip2_clsgrad_sdv14ref_rot_diff/features/GenImage__stable_diffusion_v_1_4__train.pt` (relative to script; not shipped in this repo — see §2) | Reference feature cache. |
| `--out` | `predictions.csv` | Output CSV path. |
| `--device` | `cuda:0` | Only used for on-the-fly extraction (`--image-dir`). |
| `--num-workers` | `4` | Only used for on-the-fly extraction. |
| `--model-name` | `facebook/metaclip-2-worldwide-giant` | Only used for on-the-fly extraction. |

### Output (`predictions.csv`)

| Column | Description |
|--------|--------------|
| `image_path` | Absolute path of the query image. |
| `fake_score` | Fraction of the 25 nearest reference neighbours labelled *fake* (∈ [0, 1]). |
| `prediction` | `"fake"` if `fake_score > 0.5`, else `"real"`. |
| `label` | Ground-truth label if known (0=real, 1=fake, -1=unknown), inferred from `0_real`/`1_fake` folder names or carried over from `--features`. |
| `correct` | `1` correct / `0` wrong / `-1` unknown label. |

If any rows have a known label, the script also prints overall accuracy
and ROC-AUC on those rows.

### How inference works internally

1. Load the reference cache, sub-sample 500 real + 500 fake with seed 1314,
   L2-normalise.
2. Load (or extract) query features, L2-normalise.
3. Compute cosine similarity between every query and every reference vector.
4. Take the top-25 most similar reference vectors; `fake_score` = fraction
   of those 25 labelled fake.
5. Threshold at 0.5 for the binary prediction.

## Repository layout

```
SAD-Bridge/
├── extract_features.py   # official feature extraction
├── predict.py             # official inference (kNN-25 @ layer 32)
├── requirements.txt       # Python dependencies
└── README.md              # this file
```

This repository is intentionally minimal and self-contained — no
training code, checkpoints, or datasets are included. The only external
requirement is a reference (gallery) feature cache, which you build once
with `extract_features.py` (see §2) and then reuse for all subsequent
`predict.py` calls via `--ref-cache`.
