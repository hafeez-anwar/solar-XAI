#!/usr/bin/env python3
"""
Master runner for the Model Tournament with Comprehensive Evaluation.
Aggregates: Accuracy, AUC, Inference Speed, Params, t-SNE Metrics, CV Results, and selects Winner.

"""

import argparse, sys, subprocess, json, pandas as pd
from pathlib import Path
import yaml

# Script Constants
SCRIPT_ENCODE    = "step1_encode.py"
SCRIPT_TRAIN     = "step2_svm_train.py"
SCRIPT_EXPLAIN   = "step3_explain.py"

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f: return yaml.safe_load(f)

def run_step(script_name, config_path, model_name):
    # Ensure the script name is passed correctly
    cmd = [sys.executable, script_name, "--config", str(config_path), "--model", model_name]
    print(f"\n>>> Running {script_name} for [{model_name}]...")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        print(f"!!! Error in {script_name}"); return False

def get_data(config, model_name):
    base_dir = Path(config["output"]["dir"])
    m_dir = base_dir / model_name
    
    data = {}
    # 1. Get Efficiency (Step 1)
    eff_path = m_dir / "efficiency.json"
    if eff_path.exists():
        with open(eff_path) as f: data.update(json.load(f))
        
    # 2. Get Metrics (Step 2)
    met_path = m_dir / "models" / f"{model_name}_metrics.json"
    if met_path.exists():
        with open(met_path) as f: data.update(json.load(f))
        
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        return

    cfg = load_yaml(cfg_path)
    candidates = cfg.get("models", {}).get("candidates", [])
    metric_key = cfg.get("models", {}).get("selection_metric", "accuracy")
    
    if not candidates:
        print("No candidates found in config.models.candidates")
        return

    print(f"=== Model Tournament: {candidates} ===")
    
    results_list = []
    leaderboard = {}

    # --- Phase 1 & 2: Evaluate All ---
    for model in candidates:
        print(f"\n--- Processing {model} ---")
        if not run_step(SCRIPT_ENCODE, cfg_path, model): continue
        if not run_step(SCRIPT_TRAIN, cfg_path, model): continue

        data = get_data(cfg, model)
        if data:
            score = data.get(metric_key, 0)
            leaderboard[model] = score
            print(f"--> Score ({metric_key}): {score:.4f}")
            
            results_list.append({
                "Model": model,
                "Accuracy": data.get("accuracy"),
                "AUC": data.get("auc"),
                "F1-Score": data.get("f1_macro"),
                # --- NEW COLUMNS ---
                "CV Acc (Mean)": data.get("cv_accuracy_mean"), 
                "CV Acc (Std)": data.get("cv_accuracy_std"),
                # -------------------
                "t-SNE Silhouette": data.get("tsne_silhouette"),
                "t-SNE KL-Div": data.get("tsne_kl"),
                "Params (M)": data.get("params_M"),
                "Inference (ms)": data.get("inference_ms"),
            })

    # --- Save Leaderboard ---
    if not results_list: sys.exit("No results.")
    
    df = pd.DataFrame(results_list)
    
    # Sort columns for readability (Put CV stats right after Accuracy)
    cols = list(df.columns)
    first_cols = ["Model", "Accuracy", "CV Acc (Mean)", "CV Acc (Std)", "AUC", "t-SNE Silhouette"]
    ordered_cols = [c for c in first_cols if c in cols] + [c for c in cols if c not in first_cols]
    df = df[ordered_cols]
    
    excel_path = Path(cfg["output"]["dir"]) / "model_leaderboard.xlsx"
    df.to_excel(excel_path, index=False)
    print(f"\n[SAVED] Leaderboard: {excel_path}")

    winner = max(leaderboard, key=leaderboard.get)
    print(f"\n=== WINNER: {winner} ===")

    # --- Phase 3: Explain Winner (XAI) ---
    # Runs LIME, RISE, SHAP, Rollout (and calculates Faithfulness + Runtime)
    run_step(SCRIPT_EXPLAIN, cfg_path, winner)
    
   
    print("\nTournament & Evaluation Complete.")

if __name__ == "__main__":
    main()