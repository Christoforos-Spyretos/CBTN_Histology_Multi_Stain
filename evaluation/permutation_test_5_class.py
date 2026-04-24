# %% IMPORTS
import pathlib
import itertools
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
import os
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, auc, roc_curve, f1_score

# %% UTILITIES
# set random seed for reproducibility
np.random.seed(42)

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

# %% PATH TO RESULTS
HE_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_HE_small_clam_sb_conch_v1_5.csv'
KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_KI67_small_clam_sb_conch_v1_5.csv'

EF_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Early_Fusion_HE_KI67_small_clam_sb_conch_v1_5.csv'

IM_CA_HE_inform_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CA_HE_inform_KI67_small_clam_sb_conch_v1_5.csv'
IM_CA_KI67_inform_HE_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CA_KI67_inform_HE_small_clam_sb_conch_v1_5.csv'
IM_CONCAT_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CONCAT_HE_KI67_small_clam_sb_conch_v1_5.csv'
IM_EWM_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_EWM_HE_KI67_small_clam_sb_conch_v1_5.csv'

PA_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1_5.csv'
LA_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1_5.csv'
LM_SL_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1_5.csv'
LM_OHL_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1_5.csv'
LM_THL_LM_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1_5.csv'
LM_AL_HE_KI67_conch_v1_5 = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1_5.csv'

# %% MODELS TO COMPARE
models = [
    # HE_conch_v1_5,
    KI67_conch_v1_5,

    # EF_HE_KI67_conch_v1_5,

    IM_CA_HE_inform_KI67_conch_v1_5,
    # IM_CA_KI67_inform_HE_conch_v1_5,
    # IM_CONCAT_HE_KI67_conch_v1_5,
    # IM_EWM_HE_KI67_conch_v1_5,

    # PA_HE_KI67_conch_v1_5,
    # LA_HE_KI67_conch_v1_5,
    # LM_SL_HE_KI67_conch_v1_5,
    # LM_OHL_HE_KI67_conch_v1_5,
    # "LM_THL_LM_HE_KI67_conch_v1_5",
    # "LM_AL_HE_KI67_conch_v1_5",
]


model_names = [
    # "HE_conch_v1_5",
    "KI67_conch_v1_5",

    # "EF_HE_KI67_conch_v1_5",

    "IM_CA_HE_inform_KI67_conch_v1_5",
    # "IM_CA_KI67_inform_HE_conch_v1_5",
    # "IM_CONCAT_HE_KI67_conch_v1_5",
    # "IM_EWM_HE_KI67_conch_v1_5",


    # "PA_HE_KI67_conch_v1_5",
    # "LA_HE_KI67_conch_v1_5",
    # "LM_SL_HE_KI67_conch_v1_5",
    # "LM_OHL_HE_KI67_conch_v1_5",
    # "LM_THL_LM_HE_KI67_conch_v1_5",
    # "LM_AL_HE_KI67_conch_v1_5",
]

# %% METRICS TO COMPARE
performance = []

metrics = ['BA', 'MCC', 'AUC', 'F1-Score']

for model_path in models:
    df = pd.read_csv(model_path)
    performance.append({metric: df[metric].values for metric in metrics})

# %% DOUBLE SIDED PERMUTATION TEST
print("DOUBLE SIDED PERMUTATION TEST RESULTS")
# perform pairwise permutation tests for all metrics
results_two_sided = []
num_models = len(models)
num_metrics = len(metrics)
num_comparisons = (num_models * (num_models - 1) // 2)
adjusted_alpha = 0.05 / num_comparisons  # Bonferroni correction
print(f"Adjusted alpha level after Bonferroni correction: {adjusted_alpha}")

for (i, j) in itertools.combinations(range(num_models), 2):
    model_a_name, model_b_name = model_names[i], model_names[j]
    perf_a, perf_b = performance[i], performance[j]
    for metric in metrics:
        # perform permutation test for each metric
        res = permutation_test(
            (perf_a[metric], perf_b[metric]), statistic, vectorized=True,
            permutation_type='samples', n_resamples=10000, alternative='two-sided'
        )
        results_two_sided.append({
            "Model A": model_a_name,
            "Model B": model_b_name,
            "Metric": metric,
            "Statistic": res.statistic,
            "p-value": res.pvalue,
            "Significant": res.pvalue < adjusted_alpha
        })

results_df_two_sided = pd.DataFrame(results_two_sided)
print(results_df_two_sided)

# %%