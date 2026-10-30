#!/usr/bin/env python3
"""
Pre-flight check for tournament candidates.

For every model name given, this builds the encoder with num_classes=0, runs one
forward pass at full precision and one after dynamic INT8 quantization, and prints
the resulting feature dimension. Run it before committing to a full tournament.

Two failure modes it is designed to catch:

  * an implausibly small feature dimension, which means the pooling has collapsed
    the representation, as happened with Swin when its NHWC feature map was averaged
    over the wrong axes;
  * a forward pass that raises after quantization, as happens with ConViT, whose
    gated positional self attention reaches into the nn.Linear modules that
    quantize_dynamic replaces.

Usage:
    python check_candidates.py --config config.yaml
    python check_candidates.py --models deit3_small_patch16_224 mobilevitv2_100
"""
import argparse, sys
import torch, timm, yaml


def probe(name, quantize):
    try:
        model = timm.create_model(name, pretrained=False, num_classes=0)
    except Exception as e:
        return None, f"create_model failed: {type(e).__name__}: {e}"
    model.eval()
    if quantize:
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception as e:
            return None, f"quantize_dynamic failed: {type(e).__name__}: {e}"
    try:
        cfg = timm.data.resolve_model_data_config(model)
        size = cfg.get("input_size", (3, 224, 224))[1]
    except Exception:
        size = 224
    try:
        with torch.inference_mode():
            out = model(torch.zeros(1, 3, size, size))
    except Exception as e:
        return None, f"forward failed: {type(e).__name__}: {e}"
    if out.ndim != 2:
        return None, f"forward returned shape {tuple(out.shape)}, expected (1, D)"
    return int(out.shape[-1]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str)
    ap.add_argument("--models", nargs="*", default=[])
    a = ap.parse_args()

    names = list(a.models)
    if a.config:
        with open(a.config, "r", encoding="utf-8") as f:
            names += yaml.safe_load(f).get("models", {}).get("candidates", [])
    if not names:
        sys.exit("Give --models or --config.")

    print(f"timm {timm.__version__}   torch {torch.__version__}\n")
    print(f"{'model':38s} {'FP32 dim':>9s} {'INT8 dim':>9s}  note")
    print("-" * 92)
    bad = []
    for n in dict.fromkeys(names):
        d32, e32 = probe(n, False)
        d8, e8 = probe(n, True)
        note = ""
        if e32:
            note = "FP32 " + e32
        elif e8:
            note = "INT8 " + e8
        elif d32 != d8:
            note = "feature dimension changes with precision"
        elif d32 is not None and d32 < 64:
            note = "feature dimension looks too small, check pooling"
        if note:
            bad.append(n)
        print(f"{n:38s} {str(d32 or '-'):>9s} {str(d8 or '-'):>9s}  {note}")
    print("-" * 92)
    if bad:
        print(f"\n{len(bad)} candidate(s) need attention: {', '.join(bad)}")
    else:
        print("\nAll candidates passed.")


if __name__ == "__main__":
    main()
