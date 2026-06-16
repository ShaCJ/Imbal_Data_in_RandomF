# Imbalanced Data in Random Forest Classifiers

Code for the Medium article: "Working with Imbalanced Data in Random
Forest Classifiers — a breast cancer case study".

## Setup
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate imbalanced-rf
```

## Run order
Run notebooks in order 00 → 04. Each saves its metrics to results/metrics/.
Notebook 04 loads all results and generates all figures.

## Dataset
Place Breast_Cancer.csv in the data/ folder before running.
Columns: age, race, marital_status, t_stage, n_stage, 6th_stage,
differentiate, grade, a_stage, tumor_size, estrogen_status,
progesterone_status, regional_node_examined, reginol_node_positive,
survival_months, status
