#!/usr/bin/env python3
"""Measure the on-disk footprint of each candidate encoder at FP32 and after
dynamic INT8 quantization.

Run from the folder that holds config.yaml:

    python measure_memory.py --config config.yaml --out memory_footprint.csv

The size reported is the serialized state_dict in mebibytes, which is what the
encoder occupies when it is stored or shipped to a device. Dynamic quantization
converts the weights of nn.Linear modules to 8 bit integers and leaves
convolutional and normalisation weights at full precision, so the reduction is
not a uniform factor of four and differs between architectures.
"""
import argparse, csv, io, sys
from pathlib import Path

import torch
import timm
import yaml


def state_dict_mib(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="memory_footprint.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    names = cfg["models"]["candidates"]
    print(f"{len(names)} candidates")

    rows = []
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name}", flush=True)
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0).eval()
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            fp32 = state_dict_mib(model)

            qmodel = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            int8 = state_dict_mib(qmodel)

            rows.append({
                "model": name,
                "params_M": round(params_m, 2),
                "fp32_MiB": round(fp32, 2),
                "int8_MiB": round(int8, 2),
                "reduction_MiB": round(fp32 - int8, 2),
                "reduction_pct": round(100.0 * (fp32 - int8) / fp32, 2),
                "ratio": round(fp32 / int8, 2),
            })
            print(f"      {fp32:7.2f} -> {int8:7.2f} MiB  ({rows[-1]['reduction_pct']:.1f}% smaller)")
            del model, qmodel
        except Exception as exc:
            print(f"      FAILED: {exc}", file=sys.stderr)
            rows.append({"model": name, "params_M": "", "fp32_MiB": "", "int8_MiB": "",
                         "reduction_MiB": "", "reduction_pct": "", "ratio": ""})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
