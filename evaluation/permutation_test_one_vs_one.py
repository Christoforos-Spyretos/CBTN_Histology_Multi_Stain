# %% IMPORTS
import itertools
import numpy as np
import pandas as pd
from scipy.stats import permutation_test

# %% UTILITIES
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

# %% LGG vs HGG
HE_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_small_clam_sb_uni.csv'
KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_KI67_small_clam_sb_uni.csv'
GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_GFAP_small_clam_sb_uni.csv'
EF_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_uni.csv'
EF_HE_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_uni.csv'
EF_HE_KI67_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv'
AP_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_uni.csv'
AP_HE_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_uni.csv'
AP_HE_KI67_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv'
AL_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_uni.csv'
AL_HE_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_uni.csv'
AL_HE_KI67_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv'

HE_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_small_clam_sb_conch.csv'
KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_KI67_small_clam_sb_conch.csv'
GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_GFAP_small_clam_sb_conch.csv'
EF_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch.csv'
EF_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch.csv'
EF_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
AP_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch.csv'
AP_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch.csv'
AP_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
AL_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch.csv'
AL_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_GFAP_small_clam_sb_conch.csv'
AL_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_LGG_vs_HGG_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'

# Combine both sets of models into one list
models = [
    (HE_uni, HE_conch), 
    (KI67_uni, KI67_conch), 
    (GFAP_uni, GFAP_conch),
    (EF_HE_KI67_uni, EF_HE_KI67_conch), 
    (EF_HE_GFAP_uni, EF_HE_GFAP_conch), 
    (EF_HE_KI67_GFAP_uni, EF_HE_KI67_GFAP_conch),
    (AP_HE_KI67_uni, AP_HE_KI67_conch), 
    (AP_HE_GFAP_uni, AP_HE_GFAP_conch), 
    (AP_HE_KI67_GFAP_uni, AP_HE_KI67_GFAP_conch),
    (AL_HE_KI67_uni, AL_HE_KI67_conch), 
    (AL_HE_GFAP_uni, AL_HE_GFAP_conch), 
    (AL_HE_KI67_GFAP_uni, AL_HE_KI67_GFAP_conch)
]

model_names = [
    "HE_uni vs HE_conch", 
    "KI67_uni vs KI67_conch",
    "GFAP_uni vs GFAP_conch",
    "EF_HE_KI67_uni vs EF_HE_KI67_conch",
    "EF_HE_GFAP_uni vs EF_HE_GFAP_conch",
    "EF_HE_KI67_GFAP_uni vs EF_HE_KI67_GFAP_conch",
    "AP_HE_KI67_uni vs AP_HE_KI67_conch", 
    "AP_HE_GFAP_uni vs AP_HE_GFAP_conch", 
    "AP_HE_KI67_GFAP_uni vs AP_HE_KI67_GFAP_conch",
    "AL_HE_KI67_uni vs AL_HE_KI67_conch", 
    "AL_HE_GFAP_uni vs AL_HE_GFAP_conch", 
    "AL_HE_KI67_GFAP_uni vs AL_HE_KI67_GFAP_conch"
]

# %% 5_class
HE_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_small_clam_sb_uni.csv'
KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_KI67_small_clam_sb_uni.csv'
GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_GFAP_small_clam_sb_uni.csv'
EF_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_KI67_small_clam_sb_uni.csv'
EF_HE_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_GFAP_small_clam_sb_uni.csv'
EF_HE_KI67_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv'
AP_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_KI67_small_clam_sb_uni.csv'
AP_HE_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_GFAP_small_clam_sb_uni.csv'
AP_HE_KI67_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv'
AL_HE_KI67_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_KI67_small_clam_sb_uni.csv'
AL_HE_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_GFAP_small_clam_sb_uni.csv'
AL_HE_KI67_GFAP_uni = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv'

HE_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_small_clam_sb_conch.csv'
KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_KI67_small_clam_sb_conch.csv'
GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_GFAP_small_clam_sb_conch.csv'
EF_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_KI67_small_clam_sb_conch.csv'
EF_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_GFAP_small_clam_sb_conch.csv'
EF_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
AP_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_KI67_small_clam_sb_conch.csv'
AP_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_GFAP_small_clam_sb_conch.csv'
AP_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'
AL_HE_KI67_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_KI67_small_clam_sb_conch.csv'
AL_HE_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_GFAP_small_clam_sb_conch.csv'
AL_HE_KI67_GFAP_conch = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv'

# Combine both sets of models into one list
models = [
    (HE_uni, HE_conch), 
    (KI67_uni, KI67_conch), 
    (GFAP_uni, GFAP_conch),
    (EF_HE_KI67_uni, EF_HE_KI67_conch), 
    (EF_HE_GFAP_uni, EF_HE_GFAP_conch), 
    (EF_HE_KI67_GFAP_uni, EF_HE_KI67_GFAP_conch),
    (AP_HE_KI67_uni, AP_HE_KI67_conch), 
    (AP_HE_GFAP_uni, AP_HE_GFAP_conch), 
    (AP_HE_KI67_GFAP_uni, AP_HE_KI67_GFAP_conch),
    (AL_HE_KI67_uni, AL_HE_KI67_conch), 
    (AL_HE_GFAP_uni, AL_HE_GFAP_conch), 
    (AL_HE_KI67_GFAP_uni, AL_HE_KI67_GFAP_conch)
]

model_names = [
    "HE_uni vs HE_conch", 
    "KI67_uni vs KI67_conch", 
    "GFAP_uni vs GFAP_conch",
    "EF_HE_KI67_uni vs EF_HE_KI67_conch", 
    "EF_HE_GFAP_uni vs EF_HE_GFAP_conch", 
    "EF_HE_KI67_GFAP_uni vs EF_HE_KI67_GFAP_conch",
    "AP_HE_KI67_uni vs AP_HE_KI67_conch", 
    "AP_HE_GFAP_uni vs AP_HE_GFAP_conch", 
    "AP_HE_KI67_GFAP_uni vs AP_HE_KI67_GFAP_conch",
    "AL_HE_KI67_uni vs AL_HE_KI67_conch", 
    "AL_HE_GFAP_uni vs AL_HE_GFAP_conch", 
    "AL_HE_KI67_GFAP_uni vs AL_HE_KI67_GFAP_conch"
]

# %% RUN PERMUTATION TESTS
results = []
num_comparisons = len(models)  # Each unique pair counts as one comparison
adjusted_alpha = 0.05 / num_comparisons  # Bonferroni correction
metric = 'Balanced_Accuracy' # Balanced_Accuracy

for i, (model_a_path, model_b_path) in enumerate(models):
    model_a_name, model_b_name = model_names[i].split(" vs ")
    
    df_a = pd.read_csv(model_a_path)
    df_b = pd.read_csv(model_b_path)
    
    # Extract balanced accuracy scores
    perf_a = df_a[metric].values
    perf_b = df_b[metric].values
    
    # Perform permutation test
    res = permutation_test(
        (perf_a, perf_b), statistic, vectorized=True, 
        permutation_type='samples', n_resamples=10000, alternative='two-sided'
    )
    
    results.append({
        "Model A": model_a_name,
        "Model B": model_b_name,
        "Statistic": res.statistic,
        "p-value": res.pvalue,
        "Significant": res.pvalue < adjusted_alpha
    })

results_df = pd.DataFrame(results)
print(results_df)


# Save results to a CSV file
results_df.to_csv('5_class_ba_uni_vs_conch.csv', index=False)
# %%