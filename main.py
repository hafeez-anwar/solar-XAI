#!/usr/bin/env python3
"""
Master runner for the Model Tournament with Comprehensive Evaluation.

The runner is incremental. For every candidate in the config it checks whether that
model has already been processed and skips the work that is already on disk:

  * encodings present      -> step 1 is skipped
  * metrics present        -> step 2 is skipped
  * both present           -> the model is treated as processed and only read

After every candidate has a result, the leaderboard is written and the winner is
determined from models.selection_metric. The explainability stage is expensive, so it
runs only when the winner has changed since the last run, or when its explanation
files are missing or incomplete. The winner of record is kept in winner.json next to
the leaderboard.

Use --force-encode, --force-train or --force-explain to override any of these checks.
"""

import argparse, sys, subprocess, json
from pathlib import Path
import pandas as pd
import yaml

SCRIPT_ENCODE  = "step1_encode.py"
SCRIPT_TRAIN   = "step2_svm_train.py"
SCRIPT_EXPLAIN = "step3_explain.py"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_step(script_name, config_path, model_name):
    cmd = [sys.executable, script_name, "--config", str(config_path), "--model", model_name]
    print(f"\n>>> Running {script_name} for [{model_name}]...")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        print(f"!!! Error in {script_name} for {model_name}")
        return False


# ---------- state on disk ----------

def model_dir(cfg, model):
    return Path(cfg["output"]["dir"]) / model


def has_encodings(cfg, model):
    d = model_dir(cfg, model) / "encodings"
    return (d / "features.npy").exists() and (d / "labels.npy").exists()


def has_metrics(cfg, model):
    return (model_dir(cfg, model) / "models" / f"{model}_metrics.json").exists()


def explain_enabled(cfg):
    """explain.enabled lets a pass skip the explainability stage entirely, which is
    useful for a full precision sweep whose explanations are not reported."""
    return bool(cfg.get("explain", {}).get("enabled", True))


def enabled_methods(cfg):
    m = cfg.get("explain", {}).get("methods", {}) or {}
    return [k for k, v in m.items() if v]


def explanation_state(cfg, model):
    """Returns (complete, detail). A method file that exists but holds only a header
    is reported, since attention rollout legitimately produces no rows for some
    architectures and should not force the whole stage to repeat."""
    root = Path(cfg["output"]["dir"]) / "explanations" / model
    methods = enabled_methods(cfg)
    if not methods:
        return True, "no methods enabled"
    missing, empty = [], []
    for meth in methods:
        f = root / meth / f"metrics_{meth}.csv"
        if not f.exists():
            missing.append(meth)
            continue
        try:
            rows = sum(1 for _ in f.open("r", encoding="utf-8")) - 1
        except OSError:
            rows = 0
        if rows <= 0:
            empty.append(meth)
    if missing:
        return False, f"missing: {', '.join(missing)}"
    if len(empty) == len(methods):
        return False, "all method files are empty"
    note = f"empty: {', '.join(empty)}" if empty else "complete"
    return True, note


def read_result(cfg, model):
    data = {}
    eff = model_dir(cfg, model) / "efficiency.json"
    if eff.exists():
        with eff.open() as f:
            data.update(json.load(f))
    met = model_dir(cfg, model) / "models" / f"{model}_metrics.json"
    if met.exists():
        with met.open() as f:
            data.update(json.load(f))
    return data


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--force-encode", action="store_true", help="re-encode every candidate")
    ap.add_argument("--force-train", action="store_true", help="retrain the SVM for every candidate")
    ap.add_argument("--force-explain", action="store_true", help="re-run the explainability stage")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")

    cfg = load_yaml(cfg_path)
    candidates = cfg.get("models", {}).get("candidates", []) or []
    metric_key = cfg.get("models", {}).get("selection_metric", "accuracy")
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not candidates:
        sys.exit("No candidates found in config.models.candidates")

    print(f"=== Model Tournament: {len(candidates)} candidate(s) ===")
    print(f"=== Output: {out_dir}  |  selection metric: {metric_key} ===")

    results, leaderboard, failed = [], {}, []

    for model in candidates:
        enc, met = has_encodings(cfg, model), has_metrics(cfg, model)
        if enc and met and not (args.force_encode or args.force_train):
            print(f"\n--- {model}: already processed, reading results ---")
        else:
            print(f"\n--- Processing {model} "
                  f"(encodings: {'yes' if enc else 'no'}, metrics: {'yes' if met else 'no'}) ---")
            if args.force_encode or not enc:
                if not run_step(SCRIPT_ENCODE, cfg_path, model):
                    failed.append(model); continue
            else:
                print(f"    encodings present, skipping {SCRIPT_ENCODE}")
            if args.force_encode or args.force_train or not met:
                if not run_step(SCRIPT_TRAIN, cfg_path, model):
                    failed.append(model); continue
            else:
                print(f"    metrics present, skipping {SCRIPT_TRAIN}")

        data = read_result(cfg, model)
        if not data or data.get(metric_key) is None:
            print(f"    no usable result for {model}")
            failed.append(model); continue

        leaderboard[model] = data[metric_key]
        print(f"--> {metric_key}: {leaderboard[model]:.4f}")
        results.append({
            "Model": model,
            "Accuracy": data.get("accuracy"),
            "CV Acc (Mean)": data.get("cv_accuracy_mean"),
            "CV Acc (Std)": data.get("cv_accuracy_std"),
            "AUC": data.get("auc"),
            "F1-Score": data.get("f1_macro"),
            "t-SNE Silhouette": data.get("tsne_silhouette"),
            "t-SNE KL-Div": data.get("tsne_kl"),
            "Params (M)": data.get("params_M"),
            "Inference (ms)": data.get("inference_ms"),
            "Inference IQR (ms)": data.get("inference_ms_iqr"),
            "Precision": data.get("precision"),
        })

    if not results:
        sys.exit("No results.")

    df = pd.DataFrame(results)
    first = ["Model", "Accuracy", "CV Acc (Mean)", "CV Acc (Std)", "AUC", "F1-Score", "Precision"]
    df = df[[c for c in first if c in df.columns] + [c for c in df.columns if c not in first]]
    excel_path = out_dir / "model_leaderboard.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"\n[SAVED] Leaderboard: {excel_path}")
    if failed:
        print(f"[WARN] Not included: {', '.join(failed)}")

    best = max(leaderboard.values())
    tied = [m for m in candidates if m in leaderboard and abs(leaderboard[m] - best) < 1e-9]
    winner = tied[0]
    if len(tied) > 1:
        # Prefer a tied candidate whose explanations already exist, so that models
        # sharing the same pre-trained weights do not trigger a repeat of the stage.
        done = [m for m in tied if explanation_state(cfg, m)[0]]
        if done:
            winner = done[0]
        print(f"\n=== {len(tied)} candidates tied on {metric_key}: {', '.join(tied)} ===")
    print(f"=== WINNER: {winner}  ({metric_key} = {leaderboard[winner]:.4f}) ===")

    # ---------- explainability, only when needed ----------
    state_path = out_dir / "winner.json"
    previous = None
    if state_path.exists():
        try:
            with state_path.open() as f:
                previous = json.load(f).get("model")
        except (OSError, json.JSONDecodeError):
            previous = None

    if not explain_enabled(cfg) and not args.force_explain:
        print("\n>>> Explainability stage disabled in the config (explain.enabled: false)")
        with state_path.open("w", encoding="utf-8") as f:
            json.dump({"model": winner, "selection_metric": metric_key,
                       "score": leaderboard[winner], "candidates_evaluated": len(results),
                       "explained": False}, f, indent=1)
        print("\nTournament & Evaluation Complete.")
        return

    complete, detail = explanation_state(cfg, winner)
    if args.force_explain:
        reason = "forced by --force-explain"
    elif previous != winner:
        reason = f"winner changed from {previous or 'none'} to {winner}"
    elif not complete:
        reason = f"explanations incomplete ({detail})"
    else:
        reason = None

    if reason:
        print(f"\n>>> Explainability stage: {reason}")
        ok = run_step(SCRIPT_EXPLAIN, cfg_path, winner)
        if not ok:
            print("!!! Explainability stage failed")
    else:
        print(f"\n>>> Explainability stage skipped: winner unchanged ({winner}) "
              f"and explanations are {detail}")

    with state_path.open("w", encoding="utf-8") as f:
        json.dump({"model": winner,
                   "selection_metric": metric_key,
                   "score": leaderboard[winner],
                   "candidates_evaluated": len(results)}, f, indent=1)

    print("\nTournament & Evaluation Complete.")


if __name__ == "__main__":
    main()
