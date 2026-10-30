# solar-XAI

Image based soiling detection for photovoltaic panels. A pre-trained Vision Transformer encodes
each panel image, a linear support vector machine classifies the encoding as Clean or Dusty, and the
resulting decisions are audited with LIME, RISE and SHAP against hand drawn masks.

This repository contains the code for the paper *SolCleanTransform: An Explainable Hybrid Vision
Transformer Framework for Image-based Photovoltaic Soiling Detection*.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Encode images with a candidate backbone, measure parameters and latency | `step1_encode.py` | `<model>/encodings/features.npy`, `<model>/efficiency.json` |
| Train and evaluate the linear SVM, write plots and error analysis | `step2_svm_train.py` | `<model>/models/<model>_metrics.json`, `<model>/plots/`, `<model>/error_analysis/` |
| Explain the chosen model with LIME, RISE, SHAP and attention rollout | `step3_explain.py` | `explanations/<model>/<method>/metrics_<method>.csv` |
| Run every candidate and aggregate the leaderboard | `main.py` | `model_leaderboard.xlsx` |

## Requirements

Python 3.8 or newer with `torch`, `torchvision`, `timm`, `scikit-learn`, `numpy`, `pandas`,
`matplotlib`, `seaborn`, `pyyaml`, `openpyxl`, `lime` and `shap`.

A CUDA GPU speeds up encoding but is not required. All results in the paper were produced on CPU.

## Datasets

Two sets of images are needed. Neither is stored in this repository.

### 1. Main dataset (`Dataset/`)

The images originate from the publicly available *Solar Panel Dust Detection* collection on Kaggle
(https://www.kaggle.com/datasets/pythonafroz/solar-panel-dust-detection). Because that collection 
was assembled from online sources, the same scene sometimes appears twice, once as a thumbnail and 
once at full size, and resizing every input to a fixed resolution makes such a pair effectively identical. 
The copy used in the paper has been inspected manually and the redundant images removed, 
which leaves **1,056 images, 405 Clean and 651 Dirty**.

Download the cleaned copy used in the paper:
https://drive.google.com/file/d/1b8rcGBcd71clYMl15y_c5Txwlsvpzyv8/view?usp=sharing

Extract it into a folder named `Dataset` at the project root, containing the `Clean` and `Dirty`
subfolders.

### 2. XAI ground truth (`images_GT/`)

Twenty dusty panel images with hand drawn binary masks, where white marks the soiled area. Used in
stage 3 to score localisation and faithfulness.

Download: https://drive.google.com/drive/folders/1WbJbX74HOI-3PrJrD-nBc5_oa0Z1Aqvq?usp=drive_link

Place `images_GT/` at the project root with exactly two subfolders, `images/` and `masks/`.

### Expected layout

```text
.
├── Dataset/
│   ├── Clean/          405 images
│   └── Dirty/          651 images
├── images_GT/
│   ├── images/         20 images
│   └── masks/          20 masks, suffix _mask
├── config.yaml
├── main.py
├── step1_encode.py
├── step2_svm_train.py
└── step3_explain.py
```

## Configuration

Everything is driven by `config.yaml`. The settings that change results:

| Key | Meaning |
|---|---|
| `models.candidates` | The list of `timm` model names to evaluate |
| `models.selection_metric` | Metric used to pick the model passed to stage 3. Set to `cv_accuracy_mean` so the choice is made on the training partition, matching the paper |
| `train.test_size`, `train.random_state` | Stratified holdout split, 0.2 and 42 in the paper |
| `train.C`, `train.class_weight` | Linear SVM regularisation and class weighting, 1.0 and balanced |
| `runtime.quantize` | `false` for the FP32 pass, `true` for the dynamic INT8 pass |
| `advanced.allow_fuzzy_checkpoint` | Leave `false`. When true a checkpoint is accepted on partial name overlap and a sibling variant can be loaded instead |
| `explain.enabled` | Set `false` to skip the explainability stage for this pass |
| `explain.faithfulness.mask_top_percent` | Fraction of most important pixels masked when scoring faithfulness, 0.20 in the paper |
| `explain.lime`, `explain.rise`, `explain.shap` | Per method sampling budgets, which are not equal across methods |

## Running

The runner is incremental. For each candidate it inspects what is already on disk and
repeats only what is missing:

| State found | Action |
|---|---|
| `encodings/features.npy` and `models/<model>_metrics.json` both present | model treated as processed, results read from disk |
| encodings present, metrics missing | stage 1 skipped, stage 2 run |
| neither present | both stages run |

The leaderboard is rebuilt from every processed model on each run, so
`model_leaderboard.xlsx` always reflects the full candidate list rather than only the
models touched in that invocation.

The explainability stage is the expensive one, so it runs only when it has to. After the
leaderboard is written, the winner is taken from `models.selection_metric` and compared
with the winner of record in `winner.json`. The stage runs when the winner has changed,
when its explanation files are missing or all empty, or when `--force-explain` is given.
If the winner is unchanged and its explanations are present, the stage is skipped and the
previous results stand. Set `explain.enabled: false` to skip the stage regardless, which
is appropriate for a sweep whose explanations are not reported.

Candidates that tie on the selection metric are resolved in favour of one whose
explanations already exist, so two entries that resolve to the same pre-trained weights
do not trigger a repeat.

```bash
# Full precision:  runtime.quantize: false   output.dir: "./outputs_fp32"
python main.py --config config.yaml

# Dynamic INT8:    runtime.quantize: true    output.dir: "./outputs_int8"
python main.py --config config.yaml
```

Override the checks when needed:

```bash
python main.py --config config.yaml --force-encode    # re-encode every candidate
python main.py --config config.yaml --force-train     # retrain every SVM
python main.py --config config.yaml --force-explain   # re-run the explainability stage
```

Adding a new model to `models.candidates` therefore costs only that model's encoding and
training, plus the explainability stage if the addition takes the top of the leaderboard.

Before committing to a long run, check that new candidates build, survive quantization and
produce a sensible feature dimension:

```bash
python check_candidates.py --config config.yaml
```

## Evaluation protocol

`step2_svm_train.py` creates the stratified holdout partition first, then runs five fold stratified
cross validation **on the training partition only**, and finally trains on the training partition and
evaluates once on the holdout partition. Candidate ranking therefore uses cross validation accuracy
and the holdout accuracy is reported for the selected model. 

Probabilities come from `CalibratedClassifierCV` wrapped around `LinearSVC`, so the confidence values
in `error_analysis/error_report.csv` and in the confidence histogram are Platt scaled.

## Outputs

* `efficiency.json` records `params_M` counted on the full precision graph, `inference_ms` as the
  median of 100 timed single image forward passes after ten warm-up passes, `inference_ms_iqr`,
  `precision`, and the resolved `weights` path.
* `<model>_metrics.json` records holdout accuracy, macro F1, macro precision and recall, AUC, the
  cross validation mean and standard deviation, and the t-SNE silhouette and KL divergence.
* `error_analysis/error_report.csv` lists every misclassified holdout image with its true label,
  predicted label and calibrated confidence. Comparing this file between the two passes gives the
  discordant counts for a paired McNemar test.
* `explanations/<model>/<method>/metrics_<method>.csv` gives per image IoU and Dice at thresholds
  0.3, 0.5 and 0.7, the faithfulness drop, and the runtime in seconds.

## Citation

Farman H., Anwar H., Nasralla M., Sedik A. *SolCleanTransform: An Explainable Hybrid Vision
Transformer Framework for Image-based Photovoltaic Soiling Detection.*
