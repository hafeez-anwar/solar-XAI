#!/usr/bin/env python3
"""
Step 1: Encode Images to Feature Vectors.
Optimized for CPU with Dynamic Quantization and Multi-threaded Data Loading.
"""
import argparse, csv, json, os, sys, time, fnmatch, re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# --- Performance Optimization: CPU Threading ---
# Limit threads to physical cores to prevent thrashing
try:
    import torch
    torch.set_num_threads(4) 
except ImportError:
    pass

# deps
try:
    import yaml
except Exception:
    print("Install pyyaml: pip install pyyaml", file=sys.stderr); raise
try:
    import torch.quantization
    import timm
    from PIL import Image, UnidentifiedImageError
    import numpy as np
    from numpy.lib.format import open_memmap
    from torchvision import transforms
    from torch.utils.data import Dataset, DataLoader
except Exception:
    print("Install deps: pip install torch torchvision timm pillow numpy", file=sys.stderr); raise

try:
    from safetensors.torch import load_file as safetensors_load
    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False

IMG_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------- utils ----------------
def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f: cfg = yaml.safe_load(f)
    cfg.setdefault("data", {}); cfg["data"].setdefault("exts", list(IMG_EXTS_DEFAULT))
    cfg.setdefault("output", {}); cfg.setdefault("model", {})
    cfg["model"].setdefault("name","vit_base_patch16_224"); cfg["model"].setdefault("pooling","cls"); cfg["model"].setdefault("img_size", None)
    cfg.setdefault("runtime", {}); cfg["runtime"].setdefault("batch_size", 32); cfg["runtime"].setdefault("device","cpu"); cfg["runtime"].setdefault("num_workers", 4)
    cfg.setdefault("cache", {}); cfg["cache"].setdefault("hf_hub_root", str(Path.home()/".cache"/"huggingface"/"hub"))
    cfg.setdefault("advanced", {}); cfg["advanced"].setdefault("strict_local_load", False)
    
    if "model_override" in cfg:
        cfg["model"]["name"] = cfg["model_override"]
    if not cfg["data"].get("root"): raise ValueError("config.data.root is required")
    if not cfg["output"].get("dir"): raise ValueError("config.output.dir is required")
    return cfg

def safe_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

def list_classes(root: Path) -> List[str]:
    classes = [d.name for d in root.iterdir() if d.is_dir()]
    classes.sort(key=lambda s: s.casefold())
    if not classes: raise RuntimeError(f"No class subfolders found under: {root}")
    return classes

def list_samples(root: Path, classes: List[str], exts: List[str]) -> Tuple[List[Tuple[str,int]], Dict[str,int]]:
    exts = {e.lower() for e in exts} if exts else IMG_EXTS_DEFAULT
    class_to_idx = {c:i for i,c in enumerate(classes)}
    samples: List[Tuple[str,int]] = []
    for c in classes:
        for p in (root/c).rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                rel = os.path.relpath(p, root).replace("\\","/")
                samples.append((rel, class_to_idx[c]))
    samples.sort(key=lambda x: x[0].casefold())
    if not samples: raise RuntimeError(f"No images found under: {root} (looked for {sorted(exts)})")
    return samples, class_to_idx

def build_transform(model, forced_size: Optional[int]) -> tuple:
    cfg = timm.data.resolve_model_data_config(model)
    mean = cfg.get("mean",(0.5,0.5,0.5)); std = cfg.get("std",(0.5,0.5,0.5))
    native = cfg.get("input_size",(3,224,224))[1]
    size = forced_size if (forced_size and forced_size>0) else native
    tr = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(mean,std)])
    return tr, size, mean, std

def measure_efficiency(model, device, input_size=(1, 3, 224, 224)):
    """Calculates param count and inference speed."""
    param_count = sum(p.numel() for p in model.parameters()) / 1e6 # in Millions
    dummy = torch.randn(input_size).to(device)
    
    # Temporarily disable quantization for clean speed test if needed, 
    # but strictly speaking we want to measure the ACTUAL speed (quantized).
    # So we keep it as is.
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(5): _ = model(dummy) # Warmup
            t0 = time.time()
            iters = 20
            for _ in range(iters): _ = model(dummy)
            total_time = time.time() - t0
        ms_per_image = (total_time / iters) * 1000
    except Exception as e:
        print(f"Warning: Speed test failed ({e}). Returning 0.")
        ms_per_image = 0
        
    return param_count, ms_per_image

@torch.inference_mode()
def vit_forward_embeddings(model, x: torch.Tensor, pooling: str) -> torch.Tensor:
    """
    Extracts feature embeddings. 
    Adapts to both 3D (ViT) and 4D (CNN/Hierarchical ViT) outputs.
    """
    out = model.forward_features(x)

    # 1. Handle models returning multiple stages (tuple/list)
    if isinstance(out, (list, tuple)):
        out = out[-1]

    # 2. Handle 2D: Already a flat vector (B, Num_Classes) or (B, Dim)
    if out.ndim == 2:
        return out

    # 3. Handle 3D: Standard ViT (Batch, Patches, Dim)
    if out.ndim == 3:
        if pooling == "cls":
            return out[:, 0]
        else:
            return out[:, 1:].mean(dim=1)

    # 4. Handle 4D: CNN or Hierarchical ViT (Batch, Dim, Height, Width)
    elif out.ndim == 4:
        return out.mean(dim=(-2, -1)) # Global Average Pooling

    raise RuntimeError(f"Unexpected feature shape {out.shape}")

# --- Fast Data Loading ---
class FastImageDataset(Dataset):
    def __init__(self, samples, root, transform):
        self.samples = samples
        self.root = root
        self.transform = transform
        self.dummy = torch.zeros((3, 224, 224)) # Placeholder size, resized later anyway

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, label = self.samples[idx]
        path = self.root / rel
        try:
            img = Image.open(path).convert("RGB")
            t = self.transform(img)
            return t, label, idx, rel, True # True = Success
        except Exception as e:
            # Return dummy so DataLoader doesn't crash
            return self.dummy, label, idx, f"{rel}::{str(e)}", False # False = Error

def write_csv_header_if_new(path: Path, fieldnames: List[str]):
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def read_done_set(progress_csv: Path) -> set:
    done=set()
    if progress_csv.exists():
        with progress_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status","").lower()=="done":
                    try: done.add(int(row["index"]))
                    except: pass
    return done

# --------------- HF cache ---------------
def find_torch_checkpoint(model_name: str, torch_ckpt_dir: Path) -> Optional[Path]:
    if not torch_ckpt_dir.exists(): return None
    name_tokens = [t for t in re.split(r"[_\-\.]+", model_name.lower()) if t]
    candidates = []
    for f in torch_ckpt_dir.iterdir():
        if not f.is_file(): continue
        fn = f.name.lower()
        if any(fn.endswith(ext) for ext in (".safetensors",".pth",".pt",".bin")):
            match_count = sum(1 for t in name_tokens if t in fn)
            if match_count >= 2 or model_name.lower() in fn:
                candidates.append(f)
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        return candidates[0]
    return None

def find_hf_timm_checkpoint(model_name: str, hub_root: Path) -> Optional[Path]:
    repo_dir = hub_root / f"models--timm--{model_name}"
    if not repo_dir.exists(): return None
    snap_root = repo_dir / "snapshots"
    if not snap_root.exists(): return None
    snaps = [p for p in snap_root.iterdir() if p.is_dir()]
    if not snaps: return None
    snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if HAVE_SAFETENSORS:
        for s in snaps:
            for f in s.rglob("*.safetensors"): return f
    for s in snaps:
        for f in s.rglob("*.pth"): return f
    for s in snaps:
        for pat in ["pytorch_model.bin","*.bin"]:
            for f in s.rglob("*"):
                if fnmatch.fnmatch(f.name, pat): return f
    return None

def load_local_checkpoint_into_timm(model, ckpt_path: Path, strict_local: bool) -> bool:
    try:
        if ckpt_path.suffix==".safetensors":
            if not HAVE_SAFETENSORS:
                if strict_local: raise RuntimeError("Found .safetensors but 'safetensors' not installed.")
                return False
            sd = safetensors_load(str(ckpt_path), device="cpu")
        else:
            sd = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(sd,dict) and "state_dict" in sd and isinstance(sd["state_dict"],dict): sd = sd["state_dict"]
        elif isinstance(sd,dict) and "model" in sd and isinstance(sd["model"],dict): sd = sd["model"]
        try:
            model.load_state_dict(sd, strict=True)
        except Exception:
            model.load_state_dict(sd, strict=False)
        return True
    except Exception as e:
        if strict_local: raise
        print(f"[cache] Local checkpoint load failed ({type(e).__name__}: {e}). Fallback.", file=sys.stderr)
        return False

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config-encode.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    args = parser.parse_args()
    
    cfg = load_yaml_config(Path(args.config).expanduser().resolve())
    if args.model:
        cfg["model"]["name"] = args.model

    data_root = Path(cfg["data"]["root"]).expanduser().resolve()
    project_root = Path(cfg["output"]["dir"]).expanduser().resolve()
    model_tag = safe_tag(cfg["model"]["name"])
    
    model_out_dir = project_root / model_tag 
    model_out_dir.mkdir(parents=True, exist_ok=True)
    enc_dir = model_out_dir / "encodings"
    enc_dir.mkdir(parents=True, exist_ok=True)

    enc_path = enc_dir / "features.npy"
    lab_path = enc_dir / "labels.npy"
    filenames_json = enc_dir / "filenames.json" 
    progress_csv  = enc_dir / "progress.csv"
    errors_csv    = enc_dir / "errors.csv"
    efficiency_path = model_out_dir / "efficiency.json" 

    try:
        classes = list_classes(data_root)
        samples, class_to_idx = list_samples(data_root, classes, cfg["data"]["exts"])
        N = len(samples)
    except Exception as e:
        print(f"Error scanning dataset: {e}"); sys.exit(1)

    with open(enc_dir / "classes.json", "w") as f: json.dump(class_to_idx, f, indent=2)
    with open(filenames_json, "w") as f: json.dump([s[0] for s in samples], f, indent=2)

    write_csv_header_if_new(progress_csv, ["index","relpath","status"])
    done_set = read_done_set(progress_csv)

    # Check if completely done
    if enc_path.exists() and lab_path.exists():
        try:
            labs = np.load(lab_path, mmap_mode="r")
            if labs.shape == (N,) and len(done_set) == N:
                print(f"All {N} images are already encoded for {model_tag}. Skipping.")
                return
        except Exception: pass

    # --- Load Model ---
    device_name = cfg["runtime"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available(): device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    hub_root = Path(cfg["cache"]["hf_hub_root"]).expanduser().resolve()
    torch_ckpt_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    strict_local = bool(cfg["advanced"].get("strict_local_load", False))
    
    model = None
    
    # Checkpoints
    ckpt_path = find_hf_timm_checkpoint(cfg["model"]["name"], hub_root)
    if ckpt_path:
        model = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=0)
        load_local_checkpoint_into_timm(model, ckpt_path, strict_local)
    
    if model is None:
        torch_ckpt = find_torch_checkpoint(cfg["model"]["name"], torch_ckpt_dir)
        if torch_ckpt:
            model = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=0)
            load_local_checkpoint_into_timm(model, torch_ckpt, False)

    if model is None:
        print("[cache] Downloading/Loading weights via timm...", file=sys.stderr)
        model = timm.create_model(cfg["model"]["name"], pretrained=True, num_classes=0)

    model.eval()

    # --- Optimization: Quantization ---
    if device.type == "cpu":
        print(f"⚡ Applying Dynamic Quantization for CPU speedup...")
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"Warning: Quantization failed ({e}), continuing with fp32.")

    model.to(device)
    transform, used_size, mean, std = build_transform(model, cfg["model"]["img_size"])

    # --- Efficiency Measurement ---
    print("Measuring efficiency metrics...", flush=True)
    dummy_shape = (1, 3, used_size, used_size)
    params_m, inference_ms = measure_efficiency(model, device, dummy_shape)
    print(f"  > Params: {params_m:.2f} M")
    print(f"  > Inference (Quantized): {inference_ms:.2f} ms/img")
    
    with open(efficiency_path, "w") as f:
        json.dump({"params_M": params_m, "inference_ms": inference_ms}, f)

    # Infer feature dim D
    try:
        dummy = torch.zeros(dummy_shape).to(device)
        emb = vit_forward_embeddings(model, dummy, pooling=cfg["model"]["pooling"])
        D = int(emb.shape[-1])
        print(f"Feature Dimension: {D}")
    except Exception as e:
        raise RuntimeError(f"Failed to infer feature dimension: {e}")

    # Memmaps
    mode = "r+" if enc_path.exists() else "w+"
    if mode == "w+":
        enc_mm = open_memmap(str(enc_path), mode="w+", dtype=np.float32, shape=(N, D))
        lab_mm = open_memmap(str(lab_path), mode="w+", dtype=np.int64, shape=(N,))
    else:
        enc_mm = open_memmap(str(enc_path), mode="r+", dtype=np.float32, shape=(N, D))
        lab_mm = open_memmap(str(lab_path), mode="r+", dtype=np.int64, shape=(N,))

    # --- Optimized Encoding Loop (DataLoader) ---
    batch_size = int(cfg["runtime"]["batch_size"])
    num_workers = int(cfg["runtime"]["num_workers"])
    print(f"Starting inference (Batch={batch_size}, Workers={num_workers})...")
    
    dataset = FastImageDataset(samples, data_root, transform)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    start_time = time.time()
    count = 0
    
    # Filter indices that are NOT done yet
    # Note: DataLoader goes through everything. We skip processing if done, 
    # but since random access to DataLoader is hard, we just run inference on all 
    # OR we could accept the overhead of re-inferencing some.
    # For simplicity + speed, we just run the loader. 
    # If N is huge and 90% is done, this is inefficient, but for typical use it's fine.
    
    for batch_imgs, batch_labels, batch_idxs, batch_rels, batch_valid in loader:
        
        # Filter out failed images
        valid_mask = batch_valid
        if not valid_mask.any(): continue

        # Only process valid images
        x = batch_imgs[valid_mask].to(device)
        idxs = batch_idxs[valid_mask].numpy()
        rels = np.array(batch_rels)[valid_mask.numpy()]
        lbls = batch_labels[valid_mask].numpy()
        
        # Check if we need to skip any (already done)
        # To keep batching efficient, we might process duplicates, but only write new ones.
        todo_mask = [idx not in done_set for idx in idxs]
        if not any(todo_mask): 
            continue

        try:
            feats = vit_forward_embeddings(model, x, pooling=cfg["model"]["pooling"])
            feats = feats.to(torch.float32).cpu().numpy()
            
            # Write results
            for i, idx in enumerate(idxs):
                if idx in done_set: continue
                
                enc_mm[idx, :] = feats[i]
                lab_mm[idx] = lbls[i]
                done_set.add(idx)
                count += 1
                
                # Update log
                with progress_csv.open("a", newline="", encoding="utf-8") as pf:
                    csv.writer(pf).writerow([idx, rels[i], "done"])

        except Exception as e:
            print(f"\nBatch Error: {e}")
            # Log errors for this batch
            for idx, rel in zip(idxs, rels):
                 with errors_csv.open("a", newline="", encoding="utf-8") as ef:
                    csv.writer(ef).writerow([idx, rel, str(e)])

        # Print progress
        if count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[{len(done_set)}/{N}] encoded; elapsed {elapsed:.1f}s", end="\r", flush=True)

    del enc_mm; del lab_mm
    print(f"\nDone. Processed {count} new images.")

if __name__ == "__main__":
    main()