# Pneumonia Detection from Chest X-Rays: Baseline CNN vs. Transfer Learning

Open-ended project for the **Neural Networks and Deep Learning (NNDL)** course.

This notebook classifies children's chest X-rays as **NORMAL** or **PNEUMONIA**. It compares a CNN trained
from scratch with two ImageNet-pretrained models (ConvNeXt-Tiny and EfficientNet-B0) and with two
soft-voting ensembles. All models share the same data pipeline, patient-level split, loss, optimiser and
evaluation, so any difference in results comes from the architecture and from pretraining.

## Highlights

- **Patient-level split:** train and validation are split by patient with `StratifiedGroupKFold`, using the
  patient ID in each filename, so no child appears in both sets.
- **Class imbalance:** pneumonia makes up 74% of the training images. A `WeightedRandomSampler` draws
  balanced batches; the loss stays unweighted and test metrics use the real class mix.
- **Fair thresholds:** each model's decision threshold is chosen on the validation set only (by maximising
  Youden's J), then applied unchanged to the test set.
- **Statistical checks:** McNemar tests check whether one model really beats another. Cohen's kappa
  measures how often the models disagree.
- **Reproducible:** fixed seeds, deterministic cuDNN, and seeded data-loader workers and sampler.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
by Kermany et al. on Kaggle (mirror: [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/3)).

| Folder | NORMAL | PNEUMONIA | Total |
|--------|-------:|----------:|------:|
| `train/` | 1,341 | 3,875 | 5,216 |
| `test/`  |   234 |   390 |   624 |

The dataset's own `val/` folder has only 16 images, so it is not used. Instead, `train/` is split into
**4,470 training images (2,440 patients) and 746 validation images (406 patients)**. `test/` is never used
for training or for choosing thresholds.

## Models

| Model | Pretrained | Trained layers | Trainable params | Epochs | LR |
|-------|------------|----------------|-----------------:|-------:|---:|
| Baseline CNN | none | all: 4 conv blocks (32→256 channels), global avg pool, dropout head | 1.21M | 30 | 1e-3 |
| ConvNeXt-Tiny | ImageNet | last stage + new head | 15.57M of 27.92M | 10 | 2e-4 |
| EfficientNet-B0 | ImageNet | last two blocks + new head | 1.29M of 4.17M | 10 | 2e-4 |
| Ensemble A | – | average of ConvNeXt + EfficientNet probabilities | – | – | – |
| Ensemble B | – | average of all three models' probabilities | – | – | – |

**Training setup (all models):**
- Optimiser and loss: Adam with cross-entropy loss, and `ReduceLROnPlateau` (factor 0.5, patience 2).
- Batches and input: batch size 32, 224×224 input, ImageNet normalisation.
- Augmentation: random horizontal flip and ±10° rotation, on training images only.
- Frozen layers stay in eval mode.
- Checkpoint: the checkpoint with the lowest validation loss is restored before testing.

## Results (test set, 624 images)

| Model | Threshold | Accuracy | Precision | Recall (pneumonia) | Specificity (normal) | F1 | ROC-AUC |
|-------|----------:|---------:|----------:|-------------------:|---------------------:|---:|--------:|
| Baseline CNN | 0.764 | **0.835** | **0.796** | 0.990 | **0.577** | **0.882** | 0.939 |
| ConvNeXt-Tiny | 0.492 | 0.824 | 0.783 | 0.992 | 0.543 | 0.876 | 0.969 |
| EfficientNet-B0 | 0.115 | 0.801 | 0.760 | **0.997** | 0.474 | 0.863 | 0.957 |
| Ensemble A (ConvNeXt + EfficientNet) | 0.361 | 0.819 | 0.776 | **0.997** | 0.521 | 0.873 | **0.970** |
| Ensemble B (all three) | 0.427 | 0.829 | 0.786 | **0.997** | 0.547 | 0.879 | **0.970** |

### Key findings

- **Pretraining improves ranking.** ROC-AUC is 0.957–0.970 for the pretrained models and ensembles, against
  0.939 for the CNN trained from scratch.
- **At the chosen thresholds, the baseline CNN scores best** on accuracy (83.5%) and F1 (0.882). McNemar
  tests show no significant difference between the baseline and ConvNeXt-Tiny (p = 0.37) or Ensemble A
  (p = 0.15). The baseline is significantly more accurate than EfficientNet-B0 (p = 0.004).
- **Every model catches at least 99% of pneumonia cases**, but also labels 42–53% of normal X-rays as
  pneumonia (specificity 0.47–0.58). Validation accuracy was about 98%, so this drop mostly comes from the
  known difference between the dataset's train and test folders.
- **Ensembles help very little.** The models disagree on only 6.7–7.9% of test images (Cohen's κ
  0.76–0.79), so there are few errors for averaging to cancel. The ensembles match the best ROC-AUC but
  don't beat the baseline CNN's accuracy.

## How to run

1. Clone the repo:
   ```bash
   git clone https://github.com/Yashwin69/NNDL_OPEN_CODE_NNM24AM072.git
   cd NNDL_OPEN_CODE_NNM24AM072
   ```
2. Install the dependencies. For GPU support, install the CUDA build of PyTorch from
   [pytorch.org](https://pytorch.org/get-started/locally/):
   ```bash
   pip install torch torchvision numpy pandas matplotlib scikit-learn scipy jupyter
   ```
3. Download the dataset zip from Kaggle (about 1.2 GB). Put `archive.zip` or `chest-xray-pneumonia.zip` in
   the same folder as the notebook. The first code cell extracts it into `data/`.
4. Open `NNDL_OPEN_CODE_NNM24AM072.ipynb` and click **Run All**. The first run downloads the ImageNet
   weights (about 110 MB for ConvNeXt-Tiny and 21 MB for EfficientNet-B0), so it needs an internet
   connection.

On a CUDA GPU each epoch took about 37 seconds, so all 50 epochs take roughly 30 minutes. On a CPU it will
be much slower.

## Notebook outline

1. Setup and dataset extraction
2. Dataset overview
3. Data loading, patient-level split and augmentation
4. Model 1: Baseline CNN
5. Model 2: ConvNeXt-Tiny
6. Model 3: EfficientNet-B0
7. Shared training and evaluation code
8. Training
9. Test-set evaluation: thresholds, ensembles, model disagreement
10. Comparison: training curves, confusion matrices, ROC curves, summary table, McNemar tests
11. Discussion
12. Optional: fine-tuning more of ConvNeXt-Tiny

## Limitations

- **Narrow data:** all images are of children and come from a single hospital.
- **Labels:** each label applies to one image, while the split is done by patient.
- **Thresholds:** they were tuned on only 746 validation images.
- **Not for clinical use:** this is a course project, not a diagnostic tool.

## Reference

Kermany, D. S. et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep
Learning.* Cell, 172(5).

| NNM24AM072 | Yashwin Umesha Shettigar |
