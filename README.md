# Brain-Genotype Scores: Explainable Deep Learning for Imaging Genetics

Code accompanying the paper:

> **An Explainable Deep Learning Framework for Imaging Genetics: Deriving Brain-Genotype Scores to Link Genetic Variation, Brain Structure, and Cognition**
> Kholod Thaker Alhasani, Upamanyu Ghose, Joshua Sammet, Taiyu Zhu, Sihao Xiao, Benoit Hastoy, Paul Brennan, Karen Froud, Brittany Ulm, Cornelia van Duijn, Laura M Winchester, Brian D. Marsden, Alejo Nevado-Holgado
> medRxiv preprint. https://doi.org/10.64898/2026.05.06.26352595
> Corresponding author: alejo.nevado-holgado@dpag.ox.ac.uk

## Overview

This repository contains the modelling and analysis code used to generate **brain-genotype scores**: continuous, image-based representations of genetic variation learned directly from T1-weighted structural MRI, rather than derived from predefined imaging phenotypes (IDPs).

A multi-task 3D convolutional neural network (a modified ResNet-10 backbone with Squeeze-and-Excitation blocks) was trained on T1-weighted MRI from the UK Biobank to predict genotype dosage for 120 SNPs previously associated with brain structure in GWAS. For each SNP, the network outputs a 3-class softmax probability (genotype 0/1/2 — two minor alleles, heterozygous, two major alleles); this probability distribution *is* the brain-genotype score for that SNP and subject.

These scores were then:
- **Validated** for calibration and reproducibility (expected calibration error, cross-seed correlation, dosage correlation with actual genotype).
- **Interpreted** using SmoothGrad saliency maps (via Captum) to localise the neuroanatomical regions driving each score.
- **Associated** with seven cognitive phenotypes (fluid intelligence, reaction time, numeric memory, numeric memory online, matrix pattern puzzles, symbol digit matching, tower rearranging) using nested linear regression (F-test, ΔR²), adjusted for age, sex, education, and genetic principal components.
- **Benchmarked** against classical ML models (SVM, LightGBM, logistic regression) trained on conventional IDPs, and against the actual genotype data directly. Brain-genotype scores produced 147/840 FDR-significant cognition associations, versus 1–2 total for the comparison approaches.

## Repository contents

| File | Role |
|---|---|
| `medicalnet.py` | 3D multi-task ResNet architecture (`MultiTaskResNet`), adapted from [MedicalNet](https://github.com/Tencent/MedicalNet), with Squeeze-and-Excitation channel attention. The paper's model uses the ResNet-10 configuration (`multitask_resnet10`); deeper variants are defined but unused. |
| `dataset_mn.py` | `T1_dataset`: PyTorch `Dataset` that loads T1 MRI NIfTI volumes and per-SNP genotype labels from a label CSV, applies intensity normalisation, and supports optional augmentation. |
| `loss.py` | `MaskedLoss`: multi-task cross-entropy loss that masks out missing/invalid genotype labels so each task is only trained on subjects with a valid label for that SNP. |
| `training_mn_v2.py` | Core training/evaluation logic: `trainer()` (mixed-precision training loop with early stopping, per-task balanced accuracy/F1/MCC), `tester()`, `create_saliency()` (SmoothGrad + NoiseTunnel saliency maps via Captum), `create_shap_saliency_maps()` (SHAP-based alternative explainability). |
| `main_mn.py` | Entry-point script: sets hyperparameters, loads and splits data (train/val/test), trains a model on a batch of 10 SNPs, evaluates on the held-out test set, and triggers saliency map generation. The paper's 120 SNPs were covered by 12 such runs (10 SNPs each). |
| `map_avg_2.py` | Aggregates per-subject saliency maps (from `create_saliency`) into population-level average saliency maps — overall, per predicted genotype class, and per sex — as shown in Figures 5–6 of the paper. |
| `cognition_associations_models.py` | Cognitive association analysis: nested linear regression comparing a full model (brain-genotype score probabilities + confounders) against a reduced model (confounders only), reporting the F-test p-value and ΔR² per SNP × cognitive phenotype. Produces the statistics underlying Table 4 and Figure 2. |
| `scores_calibration_assessment.ipynb` | Calibration and validation notebook: multiclass/per-class Brier score, expected calibration error (ECE), reliability diagrams, per-model and per-task calibration metrics, and dosage-correlation analysis (CNN-derived expected dosage vs. observed genotype dosage) — corresponds to the Supplementary Material's calibration section. |
| `scores_calibration_assessment_2.ipynb` | A second, condensed pass of the same calibration/dosage analysis. |

Two files referenced by the code are **not included** in this repository and would need to be supplied to run the pipeline end-to-end:
- `early_stopping.py` (defines the `EarlyStopping` class imported by `training_mn_v2.py`)
- The UK Biobank imaging and phenotype/genotype label files (see **Data availability** below)

## Pipeline

1. `main_mn.py` — for each batch of 10 SNPs, loads `dataset_mn.T1_dataset`, builds the model via `medicalnet.multitask_resnet10`, and calls `training_mn_v2.trainer()` (using `loss.MaskedLoss`) to train, then `training_mn_v2.tester()` to evaluate on the held-out test set, then `training_mn_v2.create_saliency()` to generate per-subject saliency maps.
2. `map_avg_2.py` — averages the per-subject saliency maps into the population-level maps reported in the paper.
3. `scores_calibration_assessment*.ipynb` — takes the test-set genotype probabilities produced above and assesses their calibration, reproducibility, and dosage correlation with observed genotypes.
4. `cognition_associations_models.py` — takes the same test-set brain-genotype scores and tests their association with cognitive phenotypes.

## Model architecture

3D ResNet-10 backbone (initial 7×7×7 convolution → 4 residual blocks with Squeeze-and-Excitation modules → global average pooling → 512-d feature vector), with one fully-connected classification head per SNP task, jointly trained with a summed cross-entropy objective across tasks. Input: 182×218×182 voxel, 1mm isotropic T1-weighted MRI volumes. Trained with PyTorch 1.13.1 / Python 3.10, AdamW optimiser (lr 5e-6, weight decay 8.01e-05, batch size 20, hyperparameters tuned via Optuna), on 2× NVIDIA RTX 3090 GPUs.

## Data availability

Training used T1-weighted 3D brain MRI and phenotype/genotype data from the **UK Biobank** (imaging field ID 20252, application 15181), restricted to subjects of White British ancestry. UK Biobank data are available via application: https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access. Raw imaging data, genotype/phenotype label files, and trained model weights are **not** included in this repository due to UK Biobank data governance restrictions; file paths in the code are placeholders (`/path/to/...`) to be filled in by the user with their own data locations.

## Requirements

- Python 3.10
- PyTorch 1.13.1, torchinfo
- numpy, pandas, scipy, statsmodels, scikit-learn
- nibabel, imgaug
- captum (SmoothGrad/NoiseTunnel saliency), shap
- iterstrat (`MultilabelStratifiedShuffleSplit`)
- matplotlib, seaborn
- tensorboard

## Citation

If you use this code, please cite:

```
Alhasani KT, Ghose U, Sammet J, Zhu T, Xiao S, Hastoy B, Brennan P, Froud K, Ulm B,
van Duijn C, Winchester LM, Marsden BD, Nevado-Holgado A.
An Explainable Deep Learning Framework for Imaging Genetics: Deriving Brain-Genotype
Scores to Link Genetic Variation, Brain Structure, and Cognition.
medRxiv. https://doi.org/10.64898/2026.05.06.26352595
```

## Contact

Kholod Thaker Alhasani ([KholodTAlhasani](https://github.com/KholodTAlhasani)) — Centre for Artificial Intelligence in Precision Medicines (CAIPM), King Abdulaziz University, and Centre for Medicines Discovery, University of Oxford.
Corresponding author: alejo.nevado-holgado@dpag.ox.ac.uk
