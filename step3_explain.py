#!/usr/bin/env python3
"""
Step 3: Explainability for ViT
UPDATED: Now measures and logs 'Runtime' (Computational Cost) for every image.
"""

import argparse
import csv
import math
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

# core deps
import torch
import timm
from torchvision import transforms
import matplotlib.cm as cm
from skimage.segmentation import slic

# explainers
from lime import lime_image
import shap
import joblib
import cv2

# ---------- config helpers ----------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if "output" not in cfg or "dir" not in cfg["output"]:
        raise ValueError("config.output.dir is required")
    
    if "explain" not in cfg:
        cfg["explain"] = {}

    ex = cfg["explain"]
    
    # Default Paths
    if "images_dir" not in ex:
        raise ValueError("config.explain.images_dir is required")

    if "masks_dir" not in ex:
         ex["masks_dir"] = "" 

    # defaults
    if "model" not in cfg: cfg["model"] = {}
    cfg["model"].setdefault("pooling", "cls")
    cfg["model"].setdefault("img_size", None)
    
    ex.setdefault("mask_suffix", "_mask")
    ex.setdefault("thresholds", [0.3, 0.5, 0.7])
    ex.setdefault("save_vis", True)
    ex.setdefault("overlay_alpha", 0.45)

    ex.setdefault("methods", {"rollout": True, "lime": True, "rise": True, "shap": True})
    ex.setdefault("svm", {"use_trained": True})
    
    # Faithfulness defaults
    if "faithfulness" not in ex:
        ex["faithfulness"] = {"enabled": True, "mask_top_percent": 0.20}
    ex["faithfulness"].setdefault("enabled", True)
    ex["faithfulness"].setdefault("mask_top_percent", 0.20)

    cfg.setdefault("cache", {})
    return cfg

# ---------- I/O helpers ----------
def safe_open_rgb(path: Path) -> Image.Image:
    try:
        im = Image.open(path); im.load()
        if im.mode != "RGB": im = im.convert("RGB")
        return im
    except (UnidentifiedImageError, OSError) as e:
        raise RuntimeError(f"Image load failed: {e}")

def load_mask_binary(path: Path) -> np.ndarray:
    im = Image.open(path); im.load()
    arr = np.array(im)
    if arr.ndim == 3:  
        arr = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2])
    return (arr > 0).astype(np.uint8)

def find_matching_mask(mask_dir: Path, stem: str, suffix: str) -> Optional[Path]:
    if not mask_dir or not mask_dir.exists(): return None
    for ext in IMG_EXTS:
        p = mask_dir / f"{stem}{suffix}{ext}"
        if p.exists(): return p
    for ext in IMG_EXTS:
        p = mask_dir / f"{stem}{ext}"
        if p.exists(): return p
    cands = list(mask_dir.glob(f"{stem}{suffix}.*")) or list(mask_dir.glob(f"{stem}.*"))
    return cands[0] if cands else None

def heat_overlay(img_rgb: np.ndarray, heat: np.ndarray, alpha: float=0.45) -> Image.Image:
    cmap = cm.get_cmap("jet")
    heat_rgb = (cmap(np.clip(heat,0,1))[..., :3] * 255).astype(np.uint8)
    out = (alpha * heat_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

def visualize_shap_overlay(img_rgb, heat_signed, mode="signed_heat", alpha=0.45):
    h01 = np.clip((heat_signed + 1.0) / 2.0, 0.0, 1.0)
    cmap = cm.get_cmap("seismic")
    heat_rgb = (cmap(h01)[..., :3] * 255).astype(np.uint8)
    out = (alpha * heat_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return Image.fromarray(out, "RGB")

def iou_and_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[float, float]:
    pred = pred_bin.astype(bool); gt = gt_bin.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2.0 * inter) / (pred.sum() + gt.sum() + 1e-6)
    iou = inter / (union + 1e-6)
    return float(iou), float(dice)

# ---------- Faithfulness Metric ----------
def calculate_faithfulness(img_np: np.ndarray, heat: np.ndarray, predictor, top_percent: float=0.20) -> float:
    # 1. Original Prediction
    preds_orig = predictor(img_np[None, ...])[0]
    prob_orig = preds_orig[1] 

    # 2. Create Mask
    flat = heat.flatten()
    k = int(len(flat) * (1 - top_percent))
    if k >= len(flat): k = len(flat) - 1
    thresh = np.partition(flat, k)[k]
    mask_important = (heat >= thresh)
    
    # 3. Perturb Image
    img_masked = img_np.copy()
    img_masked[mask_important] = 0 
    
    # 4. Predict on Masked
    preds_masked = predictor(img_masked[None, ...])[0]
    prob_masked = preds_masked[1]
    
    return float(prob_orig - prob_masked)

# ---------- ViT + encoder ----------
def build_transform(model, forced_size: Optional[int]):
    cfg = timm.data.resolve_model_data_config(model)
    mean = cfg.get("mean",(0.5,0.5,0.5)); std = cfg.get("std",(0.5,0.5,0.5))
    native = cfg.get("input_size",(3,224,224))[1]
    size = forced_size if (forced_size and forced_size>0) else native
    tr = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(mean,std)])
    return tr, size

@torch.inference_mode()
def vit_forward_embeddings(model, x: torch.Tensor, pooling: str) -> torch.Tensor:
    """
    Extracts feature embeddings. 
    Adapts to both 3D (ViT) and 4D (CNN/Hierarchical ViT) outputs.
    """
    out = model.forward_features(x)

    # 1. Handle models returning multiple stages (tuple/list)
    # We take the last one [-1] as it contains the highest-level semantic features.
    if isinstance(out, (list, tuple)):
        out = out[-1]

    # 2. Handle 2D: Already a flat vector (B, Num_Classes) or (B, Dim)
    if out.ndim == 2:
        return out

    # 3. Handle 3D: Standard ViT (Batch, Patches, Dim)
    if out.ndim == 3:
        # Default to CLS token (index 0). 
        # If your model doesn't use CLS, you might want out.mean(dim=1)
        if pooling == "cls":
            return out[:, 0]
        else:
            return out[:, 1:].mean(dim=1)

    # 4. Handle 4D: CNN or Hierarchical ViT (Batch, Dim, Height, Width)
    elif out.ndim == 4:
        # Global Average Pooling over H and W
        return out.mean(dim=(-2, -1))

    raise RuntimeError(f"Unexpected feature shape {out.shape}")

def try_load_vit(cfg) -> Tuple[torch.nn.Module, transforms.Compose, int]:
    model_name = cfg["model"]["name"]
    print(f"[Model] Loading: {model_name}")
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        sys.exit(1)
        
    model.eval()
    transform, used = build_transform(model, cfg["model"]["img_size"])
    return model, transform, used

# ---------- Attention Rollout ----------
@torch.inference_mode()
def attention_rollout(model, x: torch.Tensor) -> np.ndarray:
    attn_list = []
    def hook_pre(m, inputs):
        (u,) = inputs
        B, N, C = u.shape
        qkv = m.qkv(u)
        qkv = qkv.reshape(B, N, 3, m.num_heads, C // m.num_heads).permute(2,0,3,1,4)
        q, k, _ = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * m.scale
        attn = attn.softmax(dim=-1)
        attn_list.append(attn)

    handles = []
    for blk in getattr(model, "blocks", []):
        if hasattr(blk, "attn"):
            handles.append(blk.attn.register_forward_pre_hook(hook_pre))
    _ = model.forward_features(x)
    for h in handles: h.remove()

    if not attn_list: return np.zeros((x.shape[2], x.shape[3])) 

    rollout = None
    for A in attn_list:
        A = A.mean(dim=1)
        I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        A = (A + I)/2.0
        A = A / A.sum(dim=-1, keepdim=True)
        rollout = A if rollout is None else rollout @ A
    cls_to_all = rollout[:,0,:]
    return cls_to_all[:,1:]

# ---------- predictor (ViT-embedding -> SVM) ----------
class ImagePredictor:
    def __init__(self, cfg, model, transform):
        self.cfg = cfg; self.model = model; self.transform = transform
        self.use_svm = bool(cfg["explain"]["svm"].get("use_trained", True))
        self.model_name = cfg["model"]["name"]
        
        if self.use_svm:
            project_root = Path(cfg["output"]["dir"])
            models_root = project_root / self.model_name / "models"
            clf_path = models_root / f"{self.model_name}_svm.pkl"
            sca_path = models_root / f"{self.model_name}_scaler.pkl"
            
            self.clf = None
            self.scaler = None
            
            if not clf_path.exists():
                print(f"[Predictor] SVM not found, using distance.", file=sys.stderr)
                self.use_svm = False
            else:
                self.clf = joblib.load(clf_path)
                if sca_path.exists():
                    self.scaler = joblib.load(sca_path)

    @torch.inference_mode()
    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        xs = []
        for i in range(imgs.shape[0]):
            pil = Image.fromarray(imgs[i])
            xs.append(self.transform(pil))
        x = torch.stack(xs, dim=0)
        emb = vit_forward_embeddings(self.model, x, pooling=self.cfg["model"]["pooling"]).cpu().numpy()
        
        if self.use_svm:
            Z = emb
            if self.scaler is not None:
                Z = self.scaler.transform(Z).astype(np.float32, copy=False)
            if hasattr(self.clf, "predict_proba"):
                return self.clf.predict_proba(Z)
            else:
                d = self.clf.decision_function(Z).reshape(-1)
                p1 = 1.0 / (1.0 + np.exp(-d))
                p0 = 1.0 - p1
                return np.stack([p0, p1], axis=1)
        else:
            d = np.linalg.norm(emb, axis=1)
            p1 = (d - d.min()) / (d.max() - d.min() + 1e-8)
            p0 = 1.0 - p1
            return np.stack([p0, p1], axis=1)

# ---------- ROBUST Explainers ----------

def lime_explain(np_img, predictor, params):
    ns = int(params.get("num_samples", 500))
    seg = int(params.get("segments", 100))
    nf = int(params.get("num_features", 10))
    hc = params.get("hide_color", 0)

    explainer = lime_image.LimeImageExplainer()
    def predict_fn(imgs): return predictor((np.clip(imgs,0,1)*255).astype(np.uint8))
    
    exp = explainer.explain_instance(
        np_img/255.0, predict_fn, top_labels=1, hide_color=hc, num_samples=ns, 
        segmentation_fn=lambda x: slic((x*255).astype(np.uint8), n_segments=seg, compactness=10, start_label=0)
    )
    _, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=nf, hide_rest=False)
    heat = mask.astype(np.float32)
    return heat / (heat.max() + 1e-8)

def rise_explain(np_img, predictor, params):
    H, W = np_img.shape[:2]
    N = int(params.get("N", 800))
    p = float(params.get("p", 0.5))
    s_val = int(params.get("s", 7))
    sizes = params.get("sizes", [s_val])
    if not sizes: sizes = [s_val]
    jitter = int(params.get("jitter", 16))

    sal = np.zeros((H, W), dtype=np.float64)
    for _ in range(N):
        s = int(np.random.choice(sizes))
        grid = (np.random.rand(s, s) < p).astype(np.float32)
        big = cv2.resize(grid, (W + 2*jitter, H + 2*jitter), interpolation=cv2.INTER_LINEAR)
        dy, dx = np.random.randint(0, 2*jitter + 1), np.random.randint(0, 2*jitter + 1)
        mask = big[dy:dy + H, dx:dx + W][..., None]
        prob = predictor((np_img * mask).astype(np.uint8)[None, ...])[0, 1]
        sal += mask.squeeze() * prob
    sal /= (N * p + 1e-8)
    return (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

def shap_explain(np_img, predictor, params):
    K = int(params.get("superpixels", 100))
    nsamples = int(params.get("nsamples", 300))
    
    H, W = np_img.shape[:2]
    seg = slic(np_img, n_segments=K, compactness=10, start_label=0)
    bg = cv2.GaussianBlur(np_img, (0, 0), sigmaX=3)
    
    def f(z):
        imgs = np.repeat(np_img[None,...], z.shape[0], axis=0)
        for i in range(z.shape[0]):
            off = np.where(z[i]==0)[0]
            if off.size>0: 
                m = np.isin(seg, off)[...,None].astype(np.uint8)
                imgs[i] = imgs[i]*(1-m) + bg*m
        return predictor(imgs)[:, 1]
        
    expl = shap.KernelExplainer(f, np.zeros((1, int(seg.max()+1))), link="identity")
    phi = expl.shap_values(np.ones((1, int(seg.max()+1))), nsamples=nsamples)
    sv = np.array(phi[0] if isinstance(phi, list) else phi).reshape(-1)
    
    heat = np.zeros((H, W), dtype=np.float32)
    for k in range(len(sv)): heat[seg == k] = sv[k]
    return heat / (np.max(np.abs(heat)) + 1e-8)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config).expanduser().resolve())
    if args.model: cfg["model"]["name"] = args.model

    project_root = Path(cfg["output"]["dir"]).expanduser().resolve()
    model_name = cfg["model"]["name"]
    out_root = project_root / "explanations" / model_name
    out_root.mkdir(parents=True, exist_ok=True)

    model, transform, _ = try_load_vit(cfg)
    predictor = ImagePredictor(cfg, model, transform)

    img_dir = Path(cfg["explain"]["images_dir"]).expanduser().resolve()
    msk_dir = Path(cfg["explain"]["masks_dir"]).expanduser().resolve() if cfg["explain"]["masks_dir"] else None
    suffix = cfg["explain"]["mask_suffix"]
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS], key=lambda p: p.stem)
    
    methods = [m for m, on in cfg["explain"]["methods"].items() if on]
    use_faith = cfg["explain"]["faithfulness"]["enabled"]
    faith_top_p = float(cfg["explain"]["faithfulness"]["mask_top_percent"])

    for m in methods:
        print(f"\nRunning {m}...")
        base = out_root / m
        (base / "overlays").mkdir(parents=True, exist_ok=True)
        csv_path = base / f"metrics_{m}.csv"
        
        if not csv_path.exists():
            with csv_path.open("w", newline="") as f:
                # HEADER NOW INCLUDES 'runtime_sec'
                hdr = ["stem", "mask_found", "faithfulness_drop", "runtime_sec"] + sum([[f"iou@{t}", f"dice@{t}"] for t in cfg["explain"]["thresholds"]], [])
                csv.writer(f).writerow(hdr)

        for idx, img_path in enumerate(images):
            stem = img_path.stem
            if idx % 10 == 0: print(f" {idx}/{len(images)}: {stem}", end="\r")

            rgb_pil = safe_open_rgb(img_path)
            rgb_np = np.array(rgb_pil)
            
            try:
                # --- START TIMER ---
                t0 = time.time()
                
                if m == "rollout":
                    x = transform(rgb_pil).unsqueeze(0)
                    tokens = attention_rollout(model, x)
                    s = int(math.sqrt(tokens.shape[-1]))
                    heat = torch.nn.functional.interpolate(
                        tokens.reshape(1,1,s,s), size=rgb_np.shape[:2], mode="bilinear"
                    )[0,0].cpu().numpy()
                elif m == "lime": 
                    heat = lime_explain(rgb_np, predictor, cfg["explain"]["lime"])
                elif m == "rise": 
                    heat = rise_explain(rgb_np, predictor, cfg["explain"]["rise"])
                elif m == "shap": 
                    heat = shap_explain(rgb_np, predictor, cfg["explain"]["shap"])
                
                # --- END TIMER ---
                runtime = time.time() - t0

                if m == "shap":
                    vis = visualize_shap_overlay(rgb_np, heat)
                else:
                    h_disp = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
                    vis = heat_overlay(rgb_np, h_disp)
                vis.save(base / "overlays" / f"{stem}.png")

                # Faithfulness
                faith_score = 0.0
                if use_faith:
                    faith_score = calculate_faithfulness(rgb_np, heat, predictor, top_percent=faith_top_p)

                # Metrics
                mask_path = find_matching_mask(msk_dir, stem, suffix) if msk_dir else None
                # Add Runtime to Row
                row = [stem, int(bool(mask_path)), faith_score, round(runtime, 4)]
                
                if mask_path:
                    gt = load_mask_binary(mask_path)
                    if gt.shape != heat.shape:
                        gt = np.array(Image.fromarray(gt).resize((heat.shape[1], heat.shape[0]), Image.NEAREST))
                    
                    h_norm = (np.abs(heat) - np.abs(heat).min()) / (np.abs(heat).max() - np.abs(heat).min() + 1e-8)
                    
                    for t in cfg["explain"]["thresholds"]:
                        iou, dice = iou_and_dice((h_norm >= t).astype(np.uint8), gt)
                        row.extend([iou, dice])
                else:
                    row.extend(["", ""] * len(cfg["explain"]["thresholds"]))

                with csv_path.open("a", newline="") as f:
                    csv.writer(f).writerow(row)
                    
            except Exception as e:
                print(f"\nError {m} on {stem}: {e}")

    print("\nExplainability complete.")

if __name__ == "__main__":
    main()