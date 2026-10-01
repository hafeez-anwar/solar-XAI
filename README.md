# solar-XAI
# Model Tournament & Comprehensive Evaluation Pipeline

An automated machine learning pipeline for evaluating, comparing, and explaining model candidates. The system runs a "tournament" across multiple configurations, scores them against comprehensive evaluation metrics (CV Accuracy, AUC, F1, Inference Speed, t-SNE), generates an aggregated leaderboard, and automatically applies Explainable AI (XAI) techniques to the winning model.

## 🚀 Features

- **Automated Pipeline**: End-to-end execution from data encoding to model training and interpretability.
- **Comprehensive Leaderboard**: Ranks models based on Cross-Validation Accuracy, AUC, Model Parameters, Inference Runtime, and t-SNE Clustering metrics.
- **Explainable AI (XAI)**: Automatically isolates the top-performing model and applies LIME, RISE, SHAP, and Rollout to interpret its decisions, including Faithfulness and Runtime calculations.
- **Configuration-Driven**: Easily swap candidate models, input data, and selection metrics via YAML configuration.

## 💾 Datasets Configuration

This pipeline requires two sets of data to run successfully: the main training/evaluation dataset and a subset of ground truth images for Explainable AI (XAI) evaluation. 

### 1. Main Dataset (`Dataset/`)
Due to its size (GBs), the main dataset is hosted externally and is not included in this repository. 
1. Download the dataset from: [Google Drive Link](https://drive.google.com/file/d/1b8rcGBcd71clYMl15y_c5Txwlsvpzyv8/view?usp=sharing)
2. Extract it into a folder named `Dataset` at the root level. It must contain the `Clean` and `Dirty` subfolders.

### 2. XAI Ground Truth Dataset (`images_GT/`)
This dataset is used in Phase 3 to evaluate the faithfulness of the XAI methods on the winning model. 
1. Download the dataset from: [Google Drive Link](https://drive.google.com/drive/folders/1WbJbX74HOI-3PrJrD-nBc5_oa0Z1Aqvq?usp=drive_link)
2. Place the `images_GT/` folder at the root level of this project.
3. It must contain exactly two subfolders:
   - `images/`: Contains the 20 raw sample images.
   - `masks/`: Contains the 20 corresponding binary masks for those images.

### Required Folder Structure
Before running the pipeline, ensure your project directory looks exactly like this:

```text
├── Dataset/                  <-- Main dataset (Downloaded externally)
│   ├── Clean/
│   └── Dirty/
├── images_GT/                <-- XAI evaluation dataset
│   ├── images/               <-- 20 sample images
│   └── masks/                <-- 20 corresponding binary masks
├── outputs/                  <-- Auto-generated during execution
├── config-encode.yaml
├── main.py
├── step1_encode.py
├── step2_svm_train.py
└── step3_explain.py

### 🚀 How to Run the Project

Once your datasets are in place and dependencies are installed, you can run the entire pipeline from your terminal with this single command:

```bash
python main.py --config config-encode.yaml
