"""
SAD-Bridge — Official Feature Extraction Script
=================================================

Author : Xinan He
Date   : 2026-07-10
Method : SAD-Bridge (Self-consistency Anomalous-gradient Detector)

This is the OFFICIAL feature-extraction script for the SAD-Bridge model.
It extracts the SAD-Bridge CLS-gradient descriptor for a set of images using
MetaCLIP-2-worldwide-giant with the validated best configuration:

  model      : facebook/metaclip-2-worldwide-giant
  geometry   : rotation  (0° / 90° / 180° / 270°)
  loss       : MSE self-consistency across views
  grad-agg   : diff  (CLS-grad[view0] − mean(CLS-grad[view1..3]))
  layer      : 32  (key: layer_32_cls_grad)
  hidden_dim : 1664

Purpose
-------
Produces the raw (non-L2-normalised) 1664-d SAD-Bridge feature vector for
every input image. This feature is the sole input required by ``predict.py``
(kNN classification) and is also what must be extracted from a labelled
real/fake corpus to build a new reference (gallery) set.

USAGE
-----
    python SAD-Bridge/extract_features.py \\
        --image-dir /path/to/images \\
        --out features.pt \\
        --device cuda:0

    # Folder layout with labels (0_real / 1_fake sub-dirs):
    python SAD-Bridge/extract_features.py \\
        --image-dir /path/to/dataset/0_real \\
        --image-dir /path/to/dataset/1_fake \\
        --out features.pt

Output (.pt) keys
-----------------
    features     : (N, 1664) float16  — SAD-Bridge descriptors (not L2-normalised)
    image_paths  : list[str]          — absolute paths, same order as features
    labels       : (N,) int64         — 0=real, 1=fake, -1=unknown
                   Inferred from the immediate parent folder name:
                   "0_real" → 0,  "1_fake" → 1,  anything else → -1
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Fixed SAD-Bridge configuration
# ---------------------------------------------------------------------------
MODEL_NAME   = "facebook/metaclip-2-worldwide-giant"
TARGET_LAYER = 32
FEATURE_KEY  = f"layer_{TARGET_LAYER:02d}_cls_grad"

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Geometry views — rotation
# ---------------------------------------------------------------------------

def _build_rotation_views(px: torch.Tensor) -> torch.Tensor:
    """Stack 4 rotation views (0°/90°/180°/270°). Input (1,C,H,W) → (4,C,H,W)."""
    return torch.cat([TF.rotate(px, float(a)) for a in (0, 90, 180, 270)], dim=0)


# ---------------------------------------------------------------------------
# Gradient aggregation — diff
# ---------------------------------------------------------------------------

def _aggregate_diff(grad: torch.Tensor) -> np.ndarray:
    """CLS-grad[view0] − mean(CLS-grad[view1..3]).  grad: (V, seq, D)."""
    cls_g = grad[:, 0, :].detach().cpu().float()   # (V, D)
    return (cls_g[0] - cls_g[1:].mean(dim=0)).numpy()


# ---------------------------------------------------------------------------
# MSE self-consistency loss
# ---------------------------------------------------------------------------

def _geometry_loss(hidden_states, layer_idx: int) -> torch.Tensor:
    """MSE between CLS[view0] and CLS[view_i] for i=1..3, averaged."""
    cls = hidden_states[layer_idx][:, 0, :]   # (V, D)
    f0 = cls[0]
    losses = [F.mse_loss(f0, cls[i]) for i in range(1, cls.shape[0])]
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Image dataset
# ---------------------------------------------------------------------------

def _infer_label(image_path: str) -> int:
    """Infer label from immediate parent folder name."""
    parent = Path(image_path).parent.name
    if parent == "0_real":
        return 0
    if parent == "1_fake":
        return 1
    return -1


def _scan_images(paths: List[str]) -> List[str]:
    """Recursively collect image files from the given paths (files or dirs)."""
    found = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.append(str(p.resolve()))
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    found.append(str(f.resolve()))
    return found


class ImageListDataset(Dataset):
    """Load images from a flat list; label inferred from parent folder name."""

    def __init__(self, image_paths: List[str], processor):
        self.image_paths = image_paths
        self.processor   = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        path = self.image_paths[i]
        try:
            img = Image.open(path).convert("RGB")
            pv  = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception:
            pv  = self.processor(
                images=Image.new("RGB", (224, 224)), return_tensors="pt"
            )["pixel_values"].squeeze(0)
        return pv, _infer_label(path), path


def _collate(batch):
    pv     = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    paths  = [b[2] for b in batch]
    return pv, labels, paths


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _load_model(model_name: str, device: torch.device):
    try:
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32,
            attn_implementation="sdpa")
    except TypeError:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def extract_features(
    image_paths: List[str],
    model,
    processor,
    device: torch.device,
    num_workers: int = 4,
    batch_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract SAD-Bridge features for the given image paths.

    Returns
    -------
    features    : (N, 1664) float16
    labels      : (N,) int64
    valid_paths : list[str]   (skipped images are excluded)
    """
    dataset = ImageListDataset(image_paths, processor)
    loader  = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True, drop_last=False,
        collate_fn=_collate,
    )

    feat_buf:  List[np.ndarray] = []
    label_buf: List[int]        = []
    path_buf:  List[str]        = []

    for batch_px, batch_labels, batch_paths in tqdm(loader, desc="extracting"):
        for px, label, path in zip(batch_px, batch_labels, batch_paths):
            px = px.unsqueeze(0).to(device)

            views = _build_rotation_views(px)       # (4, C, H, W)
            views = views.requires_grad_(True)

            vout = model.vision_model(
                pixel_values=views, output_hidden_states=True)
            hidden_states = vout.hidden_states

            hidden_states[TARGET_LAYER].retain_grad()

            loss = _geometry_loss(hidden_states, TARGET_LAYER)
            loss.backward()

            grad = hidden_states[TARGET_LAYER].grad
            if grad is None:
                vec = np.zeros(hidden_states[TARGET_LAYER].shape[-1], dtype=np.float32)
            else:
                vec = _aggregate_diff(grad)

            feat_buf.append(vec.astype(np.float16))
            label_buf.append(label)
            path_buf.append(path)

            model.zero_grad(set_to_none=True)
            if views.grad is not None:
                views.grad = None

    if not feat_buf:
        return (np.empty((0, 1664), dtype=np.float16),
                np.empty(0, dtype=np.int64),
                [])

    features = np.stack(feat_buf, axis=0)          # (N, D) float16
    labels   = np.array(label_buf, dtype=np.int64)
    return features, labels, path_buf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAD-Bridge feature extraction.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--image-dir", type=str, action="append", dest="image_dirs",
        metavar="PATH",
        help="Image directory (or file). Repeat for multiple paths.",
    )
    p.add_argument("--out",         type=str, required=True,
                   help="Output .pt file path.")
    p.add_argument("--device",      type=str, default="cuda:0")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--model-name",  type=str, default=MODEL_NAME,
                   help="HuggingFace model name/path (default: %(default)s).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_dirs:
        raise ValueError("Provide at least one --image-dir.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {args.model_name}")

    print("\nScanning images …")
    image_paths = _scan_images(args.image_dirs)
    if not image_paths:
        raise RuntimeError("No images found in the given paths.")
    print(f"Found {len(image_paths)} images.")

    print("\nLoading model …")
    model, processor = _load_model(args.model_name, device)

    print(f"\nExtracting SAD-Bridge features  "
          f"(layer={TARGET_LAYER}, key={FEATURE_KEY}) …")
    features, labels, valid_paths = extract_features(
        image_paths, model, processor, device,
        num_workers=args.num_workers,
    )

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(
        {
            "features":    torch.from_numpy(features),
            "labels":      torch.from_numpy(labels),
            "image_paths": valid_paths,
            "model_name":  args.model_name,
            "feature_key": FEATURE_KEY,
            "layer":       TARGET_LAYER,
        },
        out_path,
    )

    n_real    = int((labels == 0).sum())
    n_fake    = int((labels == 1).sum())
    n_unknown = int((labels == -1).sum())
    print(f"\nSaved {len(valid_paths)} features → {out_path}")
    print(f"  real={n_real}  fake={n_fake}  unknown={n_unknown}")


if __name__ == "__main__":
    main()
