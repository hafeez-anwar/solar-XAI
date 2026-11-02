#!/usr/bin/env python3
"""Select the encoder from the two tournament passes.

The selection has two stages, matching the protocol described in the paper.

Stage 1, shortlist. Every candidate is ranked by cross-validation accuracy in the
full precision pass and again in the INT8 pass. Each candidate is then characterised
by the POORER of its two ranks, and the five best on that criterion form the
shortlist. Cross-validation runs on the training partition only, so no holdout image
takes part in this stage.

Stage 2, final choice. Stability and memory act as admission requirements rather than
as votes. A shortlisted candidate qualifies when its holdout accuracy moves by no more
than --max-delta images between the two precisions and quantization removes at least
--min-reduction percent of its stored size. The qualifying candidate with the highest
mean cross-validation accuracy is selected, ties within --cv-tol being broken in favour
of the smaller quantized encoder.

    python step4_select.py --fp32-dir outputs --int8-dir outputs_int8 \
        --memory memory_footprint.csv --out selection

Writes <out>_tournament.csv with every candidate and <out>.json with the shortlist
and the selected encoder.
"""
import argparse, json
from pathlib import Path

import pandas as pd
import yaml


def read_pass(root: Path, name: str) -> dict:
    mj = root / name / "models" / f"{name}_metrics.json"
    if not mj.exists():
        raise FileNotFoundError(mj)
    m = json.loads(mj.read_text())
    return {"cv": m["cv_accuracy_mean"], "cv_sd": m["cv_accuracy_std"],
            "top1": m["accuracy"], "auc": m["auc"], "f1": m["f1_macro"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fp32-dir", default="outputs")
    ap.add_argument("--int8-dir", default="outputs_int8")
    ap.add_argument("--memory", default="memory_footprint.csv")
    ap.add_argument("--out", default="selection")
    ap.add_argument("--shortlist", type=int, default=5)
    ap.add_argument("--max-delta", type=int, default=5,
                    help="largest accepted change in correctly classified images between precisions")
    ap.add_argument("--min-reduction", type=float, default=50.0,
                    help="smallest accepted reduction in stored size, percent")
    ap.add_argument("--cv-tol", type=float, default=0.01,
                    help="candidates within this much of the best mean CV accuracy are treated as equivalent")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    names = cfg["models"]["candidates"]

    rows = []
    for n in names:
        a = read_pass(Path(args.fp32_dir), n)
        b = read_pass(Path(args.int8_dir), n)
        rows.append({"model": n,
                     "cv_fp32": a["cv"], "cv_sd_fp32": a["cv_sd"], "top1_fp32": a["top1"],
                     "auc_fp32": a["auc"], "f1_fp32": a["f1"],
                     "cv_int8": b["cv"], "cv_sd_int8": b["cv_sd"], "top1_int8": b["top1"],
                     "auc_int8": b["auc"], "f1_int8": b["f1"]})
    df = pd.DataFrame(rows)

    mem = Path(args.memory)
    if mem.exists():
        df = df.merge(pd.read_csv(mem)[["model", "fp32_MiB", "int8_MiB", "reduction_pct"]],
                      on="model", how="left")
    else:
        print(f"WARNING: {mem} not found, run measure_memory.py first. "
              "Stage 2 will fall back to accuracy stability alone.")
        df["fp32_MiB"] = df["int8_MiB"] = df["reduction_pct"] = float("nan")

    # Stage 1
    df["rank_fp32"] = df["cv_fp32"].rank(ascending=False, method="min").astype(int)
    df["rank_int8"] = df["cv_int8"].rank(ascending=False, method="min").astype(int)
    df["worst_rank"] = df[["rank_fp32", "rank_int8"]].max(axis=1)
    df = df.sort_values(["worst_rank", "rank_fp32"]).reset_index(drop=True)
    short = df.head(args.shortlist).copy()

    # Stage 2. n_test is inferred from the holdout fraction so the delta is in images.
    n_test = int(round(cfg["train"].get("test_size", 0.2) * _count_images(cfg)))
    short["delta_images"] = ((short["top1_int8"] - short["top1_fp32"]) * n_test).round().astype(int)
    short["stability"] = short["delta_images"].abs()
    short["cv_mean"] = short[["cv_fp32", "cv_int8"]].mean(axis=1)

    # Stability and memory act as admission requirements, not as votes. A candidate
    # qualifies when its holdout accuracy moves by no more than --max-delta images
    # between the two precisions and quantization removes at least --min-reduction
    # percent of its stored size. Among the qualifying candidates the highest mean
    # cross-validation accuracy wins, and candidates within --cv-tol of that best
    # value are treated as equivalent and separated by the smaller INT8 encoder.
    ok = short[(short["stability"] <= args.max_delta) &
               (short["reduction_pct"].fillna(0) >= args.min_reduction)].copy()
    if ok.empty:
        raise SystemExit("No shortlisted encoder meets the stability and memory requirements. "
                         "Relax --max-delta or --min-reduction.")
    best_cv = ok["cv_mean"].max()
    tied = ok[ok["cv_mean"] >= best_cv - args.cv_tol].sort_values("int8_MiB")
    winner = tied.iloc[0]["model"]
    short["qualifies"] = short["model"].isin(ok["model"])
    short = short.sort_values(["qualifies", "cv_mean"], ascending=[False, False]).reset_index(drop=True)

    df.to_csv(f"{args.out}_tournament.csv", index=False)
    Path(f"{args.out}.json").write_text(json.dumps({
        "shortlist": short["model"].tolist(),
        "winner": winner,
        "n_test": n_test,
        "criterion": ("stage 1 poorer of the two cross-validation ranks, stage 2 highest mean "
                      "cross-validation accuracy among candidates that move by at most "
                      f"{args.max_delta} images and shed at least {args.min_reduction:g} percent of "
                      f"their stored size, ties within {args.cv_tol:g} broken by smaller INT8 size")},
        indent=2))

    cols = ["model", "cv_fp32", "cv_int8", "worst_rank", "top1_fp32", "top1_int8",
            "fp32_MiB", "int8_MiB", "reduction_pct"]
    print("\nFull field:\n", df[cols].to_string(index=False))
    print("\nShortlist:\n", short[cols + ["cv_mean", "delta_images", "qualifies"]].to_string(index=False))
    print(f"\nSelected: {winner}")


def _count_images(cfg) -> int:
    root = Path(cfg["data"]["root"]) if "root" in cfg.get("data", {}) else Path("Dataset")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in exts)


if __name__ == "__main__":
    main()
