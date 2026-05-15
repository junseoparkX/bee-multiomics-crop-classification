# Bee Multi-Omics Crop Classification

This repository contains the analysis code, processed feature sets, and manuscript-style figures for a bee multi-omics crop classification project. The study evaluates whether crop-associated molecular signatures can be recovered from tissue-specific proteomic and transcriptomic profiles, with an emphasis on temporal validation from 2020 training samples to independent 2021 test samples.

The project compares differential-abundance-derived protein candidates with sparse and model-driven feature-selection approaches, including LIMMA-based top-ranked features, STABL-selected signatures, and RFECV-based machine learning pipelines.

## Repository organization

```text
demographics/
├── Figure4/
│   ├── code/
│   └── data/
├── Figure5/
│   ├── Main_Figure_RFECV_STABL_TwoVersions_...
│   ├── Supplementary_RFECV_STABL_AllMetrics_...
│   └── Figure5_RFECV_STABL_two_main_versions_...
├── Final_Figures/
│   ├── Figure1_overall/
│   ├── MainFigure_45/
│   └── Supplementary/
├── RFECV/
│   ├── svm/
│   └── xgboost/
└── stabl/
    ├── code/
    └── data/
```

The main folder for curated output is:

```text
demographics/Final_Figures/
```

This folder contains the cleaned and organized final figures. Readers who only want to inspect the final manuscript-style outputs should start there.

## Study design

The analysis follows a temporal validation design. Feature selection and model development were performed using 2020 samples, and selected feature sets were evaluated on independent 2021 samples. This setup was used to reduce temporal leakage and to test whether crop-associated molecular signatures remain informative across years.

The main tissue-specific analyses focus on:

- Tissue A
- Tissue G
- Tissue H

The primary goal is not only to maximize classification accuracy, but also to compare feature sparsity, feature-selection agreement, and external-year robustness.

## Figure 4: LIMMA and STABL tissue-specific protein signatures

Figure 4 compares LIMMA-derived protein candidate sets with sparse STABL-selected proteomic signatures.

For the LIMMA-based feature sets, proteins were selected separately within each year and tissue using six pairwise crop contrasts:

```text
CAC_vs_CAS
CAC_vs_CRA
CAC_vs_HBB
CAS_vs_CRA
CAS_vs_HBB
CRA_vs_HBB
```

For each contrast, the top 20 non-Nosema proteins were selected primarily by LIMMA adjusted p-value (`adj.P.Val`). Raw `P.Value` was used only as a tie-breaker when available. Therefore, each tissue-year combination could contain up to 120 contrast-level candidates before duplicate protein IDs were collapsed into a final unique feature set.

STABL was used as a sparse multivariable feature-selection method to identify compact tissue-specific proteomic signatures. Figure 4 evaluates whether these smaller STABL-selected signatures preserve crop-associated structure relative to the larger LIMMA-derived candidate pools.

The figure includes:

- Heatmaps of selected tissue-specific proteomic signatures
- UMAP embeddings based on selected feature sets
- Quantitative comparison of crop-associated and location-associated separation

## Figure 5: Predictive performance, sparsity, and feature-selection agreement

Figure 5 compares STABL with RFECV-based machine learning pipelines under the same 2020-to-2021 temporal validation framework.

The RFECV analyses include:

- XGBoost + RFECV
- SVM + RFECV
- Protein-only feature sets
- Protein + RNA multi-omics feature sets

For the XGBoost + RFECV workflow, features were first screened using XGBoost feature importance to reduce the feature space to at most 30 candidates within each tissue. RFECV was then applied within this candidate set, and the final subset was selected from RFECV solutions containing 15 or fewer features based on the best internal 2020 cross-validation score.

Figure 5 summarizes:

- Micro-average AUC across tissues and omics settings
- Macro-average AUC versus selected feature count
- Feature overlap between RFECV-selected and STABL-selected features
- Cross-method consensus across STABL, XGBoost + RFECV, and SVM + RFECV

Together, these analyses evaluate whether stronger predictive discrimination is associated with compact feature selection and whether different feature-selection methods recover consistent molecular signatures.

## Methods overview

The project compares three major feature-selection strategies.

| Method | Description |
|---|---|
| LIMMA top20 combined6 | Univariate differential-abundance candidate selection across six pairwise crop contrasts |
| STABL | Sparse and stability-oriented multivariable feature selection |
| RFECV | Wrapper-based recursive feature elimination using machine learning models |

Model performance is evaluated using multiclass classification metrics, including micro-average AUC, macro-average AUC, precision, recall, and F1 score. Feature-selection agreement is evaluated using overlap and consensus analyses across methods.

## Reproducibility notes

Most notebooks are designed to be run from their corresponding analysis folders, with input files loaded from nearby `data/` folders using relative paths.

Important analysis folders include:

```text
demographics/Figure4/code/
demographics/stabl/code/
demographics/RFECV/svm/
demographics/RFECV/xgboost/
```

Final curated figures are separated from method-specific notebooks and are available in:

```text
demographics/Final_Figures/
```

## Interpretation

The analyses suggest that sparse feature-selection methods can identify compact crop-associated molecular signatures, but temporal robustness varies across tissues and methods. STABL identifies interpretable sparse protein panels and captures strong crop-associated structure in some 2020 analyses. However, RFECV-based models often show stronger performance-sparsity trade-offs when evaluated on independent 2021 samples.

These results highlight the importance of evaluating selected molecular signatures not only by within-year structure, but also by independent-year generalization.

## Repository status

This repository is organized as a manuscript-supporting analysis repository. The finalized figures are available in `demographics/Final_Figures/`, while the figure-specific and method-specific folders contain the notebooks and data files used to generate the corresponding analyses.
