#!/usr/bin/env python3
import argparse, json, sys, csv, os, shutil
from pathlib import Path
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, silhouette_score)
from sklearn.manifold import TSNE
from numpy.lib.format import open_memmap

# Ensure plots look professional
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

def load_config(path, model_override=None):
    with path.open("r") as f: cfg = yaml.safe_load(f)
    if model_override:
        cfg["model"] = {"name": model_override}
    return cfg

def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confidence_histogram(y_true, y_probs, out_path):
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    
    correct_mask = (y_pred == y_true)
    incorrect_mask = ~correct_mask
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences[correct_mask], bins=20, alpha=0.7, color='green', label='Correct')
    plt.hist(confidences[incorrect_mask], bins=20, alpha=0.7, color='red', label='Incorrect')
    plt.xlabel('Confidence Score (Probability)')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --- FIXED FUNCTION ---
def analyze_errors(y_true, y_probs, filenames, classes, out_dir, data_root):
    """
    Identifies 'Confidently Wrong' images and copies them from data_root to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure data_root is a Path object
    data_root = Path(data_root)
    
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    
    # 1. Collect Errors
    errors = []
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            errors.append({
                "filename": filenames[i],
                "true_label": classes[y_true[i]] if classes else str(y_true[i]),
                "pred_label": classes[y_pred[i]] if classes else str(y_pred[i]),
                "confidence": confidences[i]
            })
    
    # 2. Sort by Confidence (Descending) -> The "Red Bars"
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    
    # 3. Save CSV Report
    csv_path = out_dir / "error_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "pred_label", "confidence"])
        writer.writeheader()
        writer.writerows(errors)
        
    print(f"\n   [Analysis] Found {len(errors)} errors. Report saved to {csv_path}")
    
    # 4. Copy Top 10 Worst Errors for inspection
    worst_dir = out_dir / "top_confidently_wrong_images"
    worst_dir.mkdir(exist_ok=True)
    
    print(f"   [Analysis] Copying top 10 worst errors to {worst_dir}...")
    for i, err in enumerate(errors[:10]):
        # FIX: Combine data_root with the relative filename
        rel_path = err["filename"]
        src = data_root / rel_path
        
        # Name format: 0.99_Pred_Dog_True_Cat_filename.jpg
        safe_name = f"{err['confidence']:.2f}_Pred_{err['pred_label']}_True_{err['true_label']}_{Path(rel_path).name}"
        dst = worst_dir / safe_name
        
        try:
            if src.exists():
                shutil.copy(str(src), str(dst))
            else:
                print(f"     Warning: Image not found at {src}")
        except Exception as e:
            print(f"     Could not copy {src}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.model)
    model_name = cfg["model"]["name"]
    base_dir = Path(cfg["output"]["dir"])
    
    # FIX: Get the actual location of images from config
    data_root = Path(cfg["data"]["root"])
    
    model_root = base_dir / model_name
    enc_dir = model_root / "encodings"
    out_models_dir = model_root / "models"
    plots_dir = model_root / "plots"
    error_dir = model_root / "error_analysis" 
    
    out_models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Step 2: Training & Evaluation for {model_name} ===")
    
    # --- 1. Load Data ---
    enc_path = enc_dir / "features.npy"
    lab_path = enc_dir / "labels.npy"
    cls_map_path = enc_dir / "classes.json"
    files_map_path = enc_dir / "filenames.json" 
    
    if not enc_path.exists():
        print(f"Error: Features not found. Run Step 1."); sys.exit(1)

    target_names = None
    if cls_map_path.exists():
        with open(cls_map_path, "r") as f:
            class_to_idx = json.load(f)
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Load Filenames
    filenames = []
    if files_map_path.exists():
        with open(files_map_path, "r") as f:
            filenames = json.load(f)
    else:
        print("Warning: filenames.json not found. Error analysis will lack paths.")

    X = open_memmap(enc_path, mode="r")
    y = open_memmap(lab_path, mode="r")

    # --- 2. Cross-Validation ---
    print("\n>>> Running 5-Fold Cross-Validation...")
    cv_clf = LinearSVC(
        class_weight=cfg["train"].get("class_weight", None),
        C=cfg["train"].get("C", 1.0),
        max_iter=10000, 
        dual="auto"
    )
    
    # Subsample for CV if dataset is huge
    if len(y) > 50000:
        idx_cv = np.random.choice(len(y), 10000, replace=False)
        X_cv, y_cv = X[idx_cv], y[idx_cv]
    else:
        X_cv, y_cv = X, y

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_clf, X_cv, y_cv, cv=skf, scoring='accuracy', n_jobs=-1)
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"   CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # --- 3. Full Training ---
    print("\n>>> Training Final Model...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg["train"].get("test_size", 0.2), random_state=cfg["train"].get("random_state", 42))
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Get filenames for test set
    if filenames:
        filenames_test = [filenames[i] for i in test_idx]
    else:
        filenames_test = [f"Image_{i}" for i in test_idx]

    base_svc = LinearSVC(
        class_weight=cfg["train"].get("class_weight", None),
        C=cfg["train"].get("C", 1.0),
        max_iter=10000,
        dual="auto"
    )
    clf = CalibratedClassifierCV(base_svc)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)
    
    try:
        if len(target_names) == 2:
            auc = roc_auc_score(y_test, probas[:, 1])
        else:
            auc = roc_auc_score(y_test, probas, multi_class="ovr", average="macro")
    except:
        auc = 0.0

    print(f"   Test Set Results: Acc={acc:.4f}, F1={f1:.4f}")

    # --- 4. Plotting & Analysis ---
    print("   Generating Confusion Matrix...")
    cm_path = plots_dir / f"{model_name}_confusion_matrix.png"
    plot_classes = target_names if target_names else [str(i) for i in range(len(np.unique(y)))]
    plot_confusion_matrix(y_test, preds, plot_classes, cm_path)
    
    print("   Generating Confidence Histogram...")
    hist_path = plots_dir / f"{model_name}_confidence_hist.png"
    plot_confidence_histogram(y_test, probas, hist_path)

    # === FIXED: Error Analysis call with data_root ===
    print("   Analyzing Errors (The 'Red Bars')...")
    analyze_errors(y_test, probas, filenames_test, target_names, error_dir, data_root)

    print("   Generating t-SNE...")
    tsne_max = cfg["train"].get("tsne_samples", 1000)
    if len(y_test) > tsne_max:
        indices = np.random.choice(len(y_test), tsne_max, replace=False)
        X_vis, y_vis = X_test[indices], y_test[indices]
    else:
        X_vis, y_vis = X_test, y_test
        
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X_vis)
    
    try:
        kl_div = tsne.kl_divergence_
    except:
        kl_div = 0.0 # Sometimes not available in all sklearn versions
        
    sil_score = silhouette_score(X_emb, y_vis)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=y_vis, cmap='viridis', alpha=0.7)
    if target_names:
        plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
    plt.title(f"t-SNE ({model_name})\nSilhouette: {sil_score:.2f} | KL: {kl_div:.2f}")
    tsne_path = plots_dir / f"{model_name}_tsne.png"
    plt.savefig(tsne_path)
    plt.close()

    # --- 5. Save Artifacts ---
    joblib.dump(clf, out_models_dir / f"{model_name}_svm.pkl")
    
    metrics = {
        "model": str(model_name),
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "auc": float(auc),
        "cv_accuracy_mean": float(cv_mean),
        "cv_accuracy_std": float(cv_std),
        "tsne_kl": float(kl_div),
        "tsne_silhouette": float(sil_score),
        "tsne_path": str(tsne_path),
        "cm_path": str(cm_path),
        "hist_path": str(hist_path)
    }
    
    with open(out_models_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Check {error_dir} for the red-bar images!")

if __name__ == "__main__":
    main()