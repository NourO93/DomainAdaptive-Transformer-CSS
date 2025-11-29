# Cooperative Spectrum Sensing with FT-Transformer + DANN

This repository provides a clean and modular implementation of **domain adaptation for cooperative spectrum sensing** using:

- **FT-Transformer** as a tabular feature extractor  
- **DANN** (Domain-Adversarial Neural Network) for unsupervised domain alignment  
- **Semi-supervised fine-tuning** on a small labeled target subset  
- **Threshold calibration** for maximizing F1-score  

The pipeline operates on the `CASS_Spectrum_Dataset.csv` dataset and supports full training, evaluation, and fine-tuning stages.

---

## Highlights

### Domain split based on cluster SNR  
- Clusters are ranked by **average SNR across SU1 / SU2 / SU3**  
- Highest-SNR clusters → **Source domain**  
- Lowest-SNR clusters → **Target domain**

### FT-Transformer Backbone  
Uses the official `rtdl` implementation of FT-Transformer to extract expressive tabular features.

### Domain-Adversarial Training (DANN)  
- Gradient Reversal Layer (GRL)  
- **Dynamic λ** scheduling (increases from 0 → 1 during training)  
- Joint optimization of:
  - Label predictor (PU present / absent)  
  - Domain classifier (source vs. target)

### Early Stopping + Checkpointing  
Trains until validation loss on the source domain stops improving.  
Best model saved automatically to `best_dann.pt`.

### Semi-Supervised Fine-Tuning  
- Uses a **small labeled subset** of the target domain (default 5%)  
- Optional balancing of positive samples  
- Fine-tunes the whole model with a low learning rate

### Threshold Calibration  
Selects the optimal decision threshold by **maximizing F1-score** on the target test set.

### Final Evaluation Metrics  
- Accuracy  
- F1-score  
- ROC-AUC  
- Pd (Detection probability)  
- Pfa (False alarm probability)  
- Pmd (Miss detection probability)

---

## Project Structure

```text
.
├── data.py               # Data loading, preprocessing, domain splits
├── models.py             # FT-Transformer + DANN model definition
├── train_ft_dann.py      # Training, evaluation, threshold calibration
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

Dependencies include:

```
numpy
pandas
scikit-learn
torch
torchmetrics
rtdl
```

GPU support is optional but recommended.

---

## Dataset

Place your dataset in the root directory:

```
CASS_Spectrum_Dataset.csv
```

Required columns:

- `PU_Signal_Strength`  
- `Frequency_Band`  
- `SNR_SU1`, `SNR_SU2`, `SNR_SU3`  
- `Cluster_Size`  
- `Cluster_ID`  
- `Target` (binary label)

---

## Running the Pipeline

Run the entire workflow with:

```bash
python train_ft_dann.py
```

This will automatically:

1. **Load + preprocess data**  
2. **Train the DANN model**  
3. **Evaluate on target test set**  
4. **Perform semi-supervised fine-tuning** using labeled target samples  
5. **Calibrate threshold** using F1 maximization  
6. Print **final calibrated metrics**

Everything runs end-to-end without requiring configuration.

---

## Reproducibility

Training is fully deterministic via:

- `random_state=42` (sklearn)
- `seed=42` (sampling and splits)
- `TrainConfig.seed = 42`

---

## Example Output

```
===== Target Domain Evaluation (DANN) =====
Accuracy: 0.87
F1-Score: 0.84
ROC-AUC: 0.90
Pd: 0.92
Pfa: 0.08
Pmd: 0.08

===== Fine-Tuned Target Evaluation (DANN + FT + Calibration) =====
Accuracy: 0.90
F1-Score: 0.89
Pd: 0.94
Pfa: 0.06
Pmd: 0.06
```

(Values will vary depending on dataset characteristics.)


---

## Extending the Framework

This project can easily be extended to:

- Replace FT-Transformer with MLP / TabNet / hybrid models  
- Add pseudo-labeling  
- Experiment with entropy minimization  
- Use additional calibration methods (e.g., temperature scaling)

---

## Notes

This repository is fully standalone and can be integrated directly into your experiments.
