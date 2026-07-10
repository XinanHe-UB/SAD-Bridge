"""
SAD-Bridge — Official Inference Script
=========================================

Author : Xinan He
Date   : 2026-07-10
Method : SAD-Bridge (Self-consistency Anomalous-gradient Detector)

This is the OFFICIAL inference script for the SAD-Bridge model.
It classifies images as real or AI-generated using kNN-25 at layer 32 of
MetaCLIP-2-worldwide-giant with the validated SAD-Bridge configuration.

Fixed configuration (best validated settings):
  layer     : 32  (key: layer_32_cls_grad)
  knn_k     : 25
  ref_seed  : 1314  (reference sub-sample seed)
  ref_n     : 500 per class
  reference : GenImage / stable_diffusion_v_1_4 / train

Purpose
-------
Given a reference (gallery) feature cache built from a labelled real/fake
corpus (see extract_features.py) and a set of query images (or pre-extracted
query features), predicts a real/fake label + fake-probability score for
each query image via cosine-similarity kNN voting.

USAGE
-----
    # From pre-extracted features (recommended):
    python SAD-Bridge/predict.py \\
        --features query.pt \\
        --ref-cache /path/to/checkpoint/features/GenImage__stable_diffusion_v_1_4__train.pt \\
        --out predictions.csv

    # Extract on-the-fly and predict:
    python SAD-Bridge/predict.py \\
        --image-dir /path/to/images \\
        --ref-cache /path/to/checkpoint/features/GenImage__stable_diffusion_v_1_4__train.pt \\
        --out predictions.csv \\
        --device cuda:0

Default --ref-cache path
------------------------
    checkpoints/metaclip2_clsgrad_sdv14ref_rot_diff/features/
        GenImage__stable_diffusion_v_1_4__train.pt
    (resolved relative to the script's parent directory)

Output CSV columns
------------------
    image_path   : absolute path to the query image
    fake_score   : fraction of k=25 nearest neighbours labelled fake  [0, 1]
    prediction   : "fake" if fake_score > 0.5, else "real"
    label        : ground-truth label (0=real / 1=fake / -1=unknown)
                   when available from folder structure or --features file
    correct      : 1/0/-1  (−1 when label is unknown)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Fixed SAD-Bridge configuration
# ---------------------------------------------------------------------------
TARGET_LAYER = 32
FEATURE_KEY  = f"layer_{TARGET_LAYER:02d}_cls_grad"
KNN_K        = 25
REF_SEED     = 1314
REF_N        = 500          # max samples per class from reference

_DEFAULT_REF_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "checkpoints",
    "metaclip2_clsgrad_sdv14ref_rot_diff",
    "features",
    "GenImage__stable_diffusion_v_1_4__train.pt",
)


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def _load_reference(ref_cache: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load reference features from the pre-extracted cache .pt file.

    Returns L2-normalised (N_ref, D) float32 features and (N_ref,) int64 labels.
    Sub-samples REF_N per class with seed REF_SEED (matching training best config).
    """
    if not os.path.exists(ref_cache):
        raise FileNotFoundError(
            f"Reference cache not found: {ref_cache}\n"
            "Run extract_features.py on the SDv1.4 train split, or point\n"
            "--ref-cache to the existing checkpoint features directory."
        )
    d = torch.load(ref_cache, map_location="cpu")

    # Support both cache formats:
    #   (a) new format: d["features"], d["labels"]  (from extract_features.py)
    #   (b) old format: d[FEATURE_KEY], d["labels"]  (from eval script)
    if "features" in d:
        feats = d["features"]
    elif FEATURE_KEY in d:
        feats = d[FEATURE_KEY]
    else:
        raise KeyError(
            f"Neither 'features' nor '{FEATURE_KEY}' found in {ref_cache}.\n"
            f"Available keys: {[k for k in d if not k.startswith('_')]}"
        )

    labels = d["labels"]
    if isinstance(feats, torch.Tensor):
        feats = feats.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    feats  = feats.astype(np.float32)
    labels = labels.astype(np.int64)

    # Sub-sample REF_N per class with fixed seed
    rng = np.random.default_rng(REF_SEED)
    idx_buf, lbl_buf = [], []
    for lbl in (0, 1):
        idx = np.flatnonzero(labels == lbl)
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        idx = idx[:REF_N]
        idx_buf.append(idx)
        lbl_buf.append(np.full(len(idx), lbl, dtype=np.int64))
    if not idx_buf:
        raise RuntimeError("Reference cache contains no labelled samples.")

    sel_idx    = np.concatenate(idx_buf)
    sel_labels = np.concatenate(lbl_buf)
    sel_feats  = feats[sel_idx]

    n_ref = sel_feats.shape[0]
    norms = np.linalg.norm(sel_feats, axis=-1, keepdims=True)
    sel_feats = sel_feats / (norms + 1e-8)

    n_real = int((sel_labels == 0).sum())
    n_fake = int((sel_labels == 1).sum())
    print(f"Reference: {n_ref} samples  (real={n_real}, fake={n_fake})")
    return sel_feats, sel_labels


# ---------------------------------------------------------------------------
# kNN inference
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x.astype(np.float32), axis=-1, keepdims=True)
    return x.astype(np.float32) / (n + 1e-8)


def knn_predict(
    query_feats: np.ndarray,
    ref_feats: np.ndarray,
    ref_labels: np.ndarray,
    k: int = KNN_K,
    batch: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    kNN classification.

    Parameters
    ----------
    query_feats : (N, D) float32 — already L2-normalised or will be normalised
    ref_feats   : (M, D) float32 — already L2-normalised
    ref_labels  : (M,) int64

    Returns
    -------
    fake_scores : (N,) float32 — fraction of k nearest neighbours labelled fake
    predictions : (N,) int64  — 0=real, 1=fake  (threshold 0.5)
    """
    query_feats = _l2_normalize(query_feats)
    q_t   = torch.from_numpy(query_feats).float()
    r_t   = torch.from_numpy(ref_feats).float()
    lbl_t = torch.from_numpy(ref_labels).long()

    if torch.cuda.is_available():
        r_t   = r_t.cuda()
        lbl_t = lbl_t.cuda()

    scores_buf: List[float] = []
    for start in tqdm(range(0, len(q_t), batch), desc="kNN", leave=False):
        chunk = q_t[start : start + batch]
        if torch.cuda.is_available():
            chunk = chunk.cuda(non_blocking=True)
        sim           = chunk @ r_t.T                    # (B, M)
        topk_vals, topk_idx = sim.topk(k, dim=-1)        # (B, k)
        nb_labels     = lbl_t[topk_idx]                  # (B, k)
        score         = nb_labels.float().mean(dim=-1)   # (B,)
        scores_buf.extend(score.detach().cpu().tolist())

    fake_scores = np.array(scores_buf, dtype=np.float32)
    predictions = (fake_scores > 0.5).astype(np.int64)
    return fake_scores, predictions


# ---------------------------------------------------------------------------
# Query feature loading
# ---------------------------------------------------------------------------

def _load_query_features(
    features_path: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load query features from a .pt file produced by extract_features.py."""
    d = torch.load(features_path, map_location="cpu")
    feats = d["features"]
    if isinstance(feats, torch.Tensor):
        feats = feats.numpy()
    feats = feats.astype(np.float32)

    labels = d.get("labels", None)
    if labels is None:
        labels = np.full(len(feats), -1, dtype=np.int64)
    elif isinstance(labels, torch.Tensor):
        labels = labels.numpy().astype(np.int64)

    paths = d.get("image_paths", [f"image_{i}" for i in range(len(feats))])
    return feats, labels, paths


# ---------------------------------------------------------------------------
# On-the-fly extraction (imports from extract_features.py in same dir)
# ---------------------------------------------------------------------------

def _extract_on_the_fly(
    image_dirs: List[str],
    device: torch.device,
    model_name: str,
    num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_features import _load_model, extract_features, _scan_images

    print(f"\nLoading model: {model_name} …")
    model, processor = _load_model(model_name, device)

    image_paths = _scan_images(image_dirs)
    if not image_paths:
        raise RuntimeError("No images found.")
    print(f"Found {len(image_paths)} images.")

    feats, labels, valid_paths = extract_features(
        image_paths, model, processor, device, num_workers=num_workers,
    )
    feats = feats.astype(np.float32)
    return feats, labels, valid_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAD-Bridge inference (kNN-25, layer 32).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--features", type=str, metavar="PATH",
        help=".pt file produced by extract_features.py",
    )
    src.add_argument(
        "--image-dir", type=str, action="append", dest="image_dirs",
        metavar="PATH",
        help="Image directory for on-the-fly extraction (repeat for multiple paths).",
    )
    p.add_argument(
        "--ref-cache", type=str, default=_DEFAULT_REF_CACHE,
        metavar="PATH",
        help="Path to the reference feature cache .pt file.\n"
             f"Default: {_DEFAULT_REF_CACHE}",
    )
    p.add_argument("--out",         type=str, default="predictions.csv",
                   help="Output CSV path (default: predictions.csv).")
    p.add_argument("--device",      type=str, default="cuda:0",
                   help="Device for on-the-fly extraction (ignored with --features).")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--model-name",  type=str,
                   default="facebook/metaclip-2-worldwide-giant")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -- Load reference --------------------------------------------------
    print(f"Loading reference: {args.ref_cache}")
    ref_feats, ref_labels = _load_reference(args.ref_cache)

    # -- Load / extract query features -----------------------------------
    if args.features:
        print(f"\nLoading query features: {args.features}")
        query_feats, query_labels, image_paths = _load_query_features(args.features)
    else:
        device = torch.device(
            args.device if torch.cuda.is_available() else "cpu")
        query_feats, query_labels, image_paths = _extract_on_the_fly(
            args.image_dirs, device, args.model_name, args.num_workers,
        )

    print(f"\nQuery: {len(query_feats)} images")

    # -- kNN inference ---------------------------------------------------
    print(f"\nRunning kNN-{KNN_K} at layer {TARGET_LAYER} …")
    fake_scores, predictions = knn_predict(query_feats, ref_feats, ref_labels)

    # -- Build output ----------------------------------------------------
    df = pd.DataFrame({
        "image_path": image_paths,
        "fake_score": fake_scores.round(4),
        "prediction": ["fake" if p == 1 else "real" for p in predictions],
        "label":      query_labels,
    })

    # correctness: 1=correct, 0=wrong, -1=unknown
    def _correct(row):
        if row["label"] == -1:
            return -1
        return int(row["label"] == (1 if row["prediction"] == "fake" else 0))
    df["correct"] = df.apply(_correct, axis=1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nPredictions saved → {args.out}")

    # -- Summary ---------------------------------------------------------
    n_fake = int((predictions == 1).sum())
    n_real = int((predictions == 0).sum())
    print(f"  Predicted  real={n_real}  fake={n_fake}")

    known = df[df["label"] != -1]
    if len(known) > 0:
        acc = (known["correct"] == 1).mean()
        print(f"  Accuracy on labelled images: {acc:.4f}  (N={len(known)})")
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(known["label"].values,
                                known["fake_score"].values)
            print(f"  AUC on labelled images: {auc:.4f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
