# Unbiased-Toxicity-Detection-in-Online-Comments

**Notebook:** `Unbiased Toxicity.ipynb`
**Objective:** Train and evaluate a toxicity classifier that is fair across identity subgroups, with emphasis on worst-group performance and bias-aware evaluation.

## Overview

This notebook:

* Loads a Jigsaw-style toxicity dataset with identity columns and toxicity labels.
* Trains baseline and transformer-based models (optionally Detoxify) on comment text.
* Computes fairness metrics such as Worst-Group Accuracy, Subgroup AUC, BPSN AUC, and BNSP AUC.
* Supports class/instance reweighting, bias-aware loss options, calibration, and threshold tuning.
* Produces tables and plots for overall and per-group performance.

---

## Requirements

Python 3.9–3.11 recommended.

```
pandas
numpy
scikit-learn
scipy
torch
transformers
datasets
torchmetrics
evaluate
imbalanced-learn
tqdm
matplotlib
seaborn
detoxify           # optional (UnitaryAI Detoxify models)
python-dotenv      # optional (for local secrets)
```

Installation example (CPU):

```bash
pip install -U pandas numpy scikit-learn scipy torch transformers datasets \
  torchmetrics evaluate imbalanced-learn tqdm matplotlib seaborn detoxify python-dotenv
```

For GPU acceleration, install the CUDA build of PyTorch that matches your system (see pytorch.org).

---

## Data Schema

Assumed columns (customize in the setup cell if yours differ):

* Text column: `comment_text` (or `text`)
* Target(s): either a single `toxicity` label/score or multi-label targets such as `toxic`, `insult`, `threat`, `obscene`, etc.
* Identity columns (binary flags):
  `male, female, LGBTQ, christian, muslim, other_religions, black, white`

Configure names in the notebook:

```python
TEXT_COL = "comment_text"
TARGET_COLS = ["toxicity"]  # or ["toxic","insult","threat","obscene",...]
IDENTITY_COLS = ["male","female","LGBTQ","christian","muslim","other_religions","black","white"]
```

---

## How to Run

### Google Colab

1. Upload the notebook and your dataset CSV(s).
2. Enable GPU (Runtime → Change runtime type).
3. Set column names and file paths in the configuration cell.
4. Run cells top to bottom.

### Local (Jupyter or VS Code)

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt   # or use the pip line above
jupyter lab                   # or: jupyter notebook
```

Open `Unbiased Toxicity.ipynb` and execute cells sequentially.

---

## Models

* Baselines: TF-IDF + Logistic Regression or Linear SVM.
* Transformer models: e.g., Detoxify (unbiased) or a RoBERTa/BERT checkpoint via `transformers`.
* Optional techniques: class weighting, focal loss, label smoothing.
* Bias mitigation: group reweighting, per-group thresholds, calibration.

---

## Metrics and Bias Evaluation

Reported both overall and per subgroup.

* Worst-Group Accuracy (or min-F1/min-recall): minimum performance across identity subgroups.
* Subgroup AUC: AUC computed within each identity subgroup.
* BPSN AUC (Background Positive, Subgroup Negative): AUC on background positives vs subgroup negatives.
* BNSP AUC (Background Negative, Subgroup Positive): AUC on background negatives vs subgroup positives.
* Optional fairness views: Equality of Opportunity gaps (TPR differences across groups).

Thresholding:

* Global threshold tuned on validation ROC/PR.
* Optional per-group thresholds to equalize or target specific TPR/FNR trade-offs.

---

## Outputs

Configurable output directory (see “Outputs” cell). Typical artifacts:

```
./outputs/
  metrics_overall.csv
  metrics_per_group.csv
  preds_with_groups.csv
  plots/
    roc_per_group.png
    pr_per_group.png
    confusion_by_group.png
    calibration_per_group.png
```

---

## Reproducibility

* Set global seeds for `random`, `numpy`, and `torch`.
* Use stratified splits or GroupKFold where appropriate.
* Log package versions for the run.

---

## Suggested Workflow

1. Train a fast baseline and compare overall vs per-group metrics.
2. Fine-tune a transformer model and compare against the baseline on Worst-Group Accuracy and subgroup AUCs.
3. Apply bias mitigation techniques:

   * Group reweighting for underrepresented or underperforming groups.
   * Focal loss for heavy imbalance.
   * Per-group thresholding and calibration.
4. Select the model using Worst-Group Accuracy or a composite fairness-aware metric.

---

## Notes and Gotchas

* Identity sparsity: use confidence intervals when comparing small groups.
* Multi-label setups: compute metrics per label and aggregate.
* Text truncation: ensure max sequence length is large enough; log truncation rates.
* Threshold tuning should be validated on a held-out set to avoid optimistic bias.

---

## Acknowledgments

* Jigsaw Unintended Bias in Toxicity Classification dataset
* UnitaryAI Detoxify models
* scikit-learn, transformers, torchmetrics, and related tooling

---

## License

For academic or educational use. Ensure ethical handling of identity-labeled data and follow dataset terms.
