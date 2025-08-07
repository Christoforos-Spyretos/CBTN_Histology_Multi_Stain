# %% IMPORTS
import pandas as pd

# %% LOAD CSV
HE_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_small_clam_sb_uni.csv')
KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_KI67_small_clam_sb_uni.csv')
GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_GFAP_small_clam_sb_uni.csv')
EF_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_KI67_small_clam_sb_uni.csv')
EF_HE_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_GFAP_small_clam_sb_uni.csv')
EF_HE_KI67_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv')
AP_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_KI67_small_clam_sb_uni.csv')
AP_HE_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_GFAP_small_clam_sb_uni.csv')
AP_HE_KI67_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv')
AG_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_KI67_small_clam_sb_uni.csv')
AG_HE_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_GFAP_small_clam_sb_uni.csv')
AG_HE_KI67_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv')
MAJ_VOT_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_5_class_Merged_HE_KI67_small_clam_sb_uni.csv')
MAJ_VOT_HE_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_5_class_Merged_HE_GFAP_small_clam_sb_uni.csv')
MAJ_VOT_HE_KI67_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_5_class_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv')
Simple_MLP_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_5_class_Merged_HE_KI67_small_clam_sb_uni.csv')
Simple_MLP_HE_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_5_class_Merged_HE_GFAP_small_clam_sb_uni.csv')
Simple_MLP_HE_KI67_GFAP_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_5_class_Merged_HE_KI67_GFAP_small_clam_sb_uni.csv')

HE_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_small_clam_sb_conch.csv')
KI67_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_KI67_small_clam_sb_conch.csv')
GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_GFAP_small_clam_sb_conch.csv')
EF_HE_KI67_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_KI67_small_clam_sb_conch.csv')
EF_HE_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_GFAP_small_clam_sb_conch.csv')
EF_HE_KI67_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv')
AP_HE_KI67_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_KI67_small_clam_sb_conch.csv')
AP_HE_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_GFAP_small_clam_sb_conch.csv')
AP_HE_KI67_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_prob_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv')
AG_HE_KI67_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_KI67_small_clam_sb_conch.csv')
AG_HE_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_GFAP_small_clam_sb_conch.csv')
AG_HE_KI67_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_agg_logits_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv')
MAJ_VOT_HE_KI67_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_5_class_Merged_HE_KI67_small_clam_sb_conch.csv')
MAJ_VOT_HE_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_5_class_Merged_HE_GFAP_small_clam_sb_conch.csv')
MAJ_VOT_HE_KI67_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_majority_voting_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv')
Simple_MLP_HE_KI67_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_5_class_Merged_HE_KI67_small_clam_sb_conch.csv')
Simple_MLP_HE_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_5_class_Merged_HE_GFAP_small_clam_sb_conch.csv')
Simple_MLP_HE_KI67_GFAP_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Late_Fusion_simple_model_mlp_5_class_Merged_HE_KI67_GFAP_small_clam_sb_conch.csv')

# %% AGGREGATE CSV FILES
# Aggregate all models
summary = pd.concat([
    HE_uni, 
    KI67_uni, 
    GFAP_uni,
    HE_conch, 
    KI67_conch,
    GFAP_conch,
    
    EF_HE_KI67_uni, 
    EF_HE_GFAP_uni, 
    EF_HE_KI67_GFAP_uni,
    EF_HE_KI67_conch,
    EF_HE_GFAP_conch,
    EF_HE_KI67_GFAP_conch,
    
    AP_HE_KI67_uni, 
    AP_HE_GFAP_uni, 
    AP_HE_KI67_GFAP_uni,
    AP_HE_KI67_conch,
    AP_HE_GFAP_conch,
    AP_HE_KI67_GFAP_conch,
    
    AG_HE_KI67_uni, 
    AG_HE_GFAP_uni, 
    AG_HE_KI67_GFAP_uni,
    AG_HE_KI67_conch,
    AG_HE_GFAP_conch,
    AG_HE_KI67_GFAP_conch,

    MAJ_VOT_HE_KI67_uni,
    MAJ_VOT_HE_GFAP_uni,
    MAJ_VOT_HE_KI67_GFAP_uni,
    MAJ_VOT_HE_KI67_conch,
    MAJ_VOT_HE_GFAP_conch,
    MAJ_VOT_HE_KI67_GFAP_conch,

    Simple_MLP_HE_KI67_uni,
    Simple_MLP_HE_GFAP_uni,
    Simple_MLP_HE_KI67_GFAP_uni,
    Simple_MLP_HE_KI67_conch,
    Simple_MLP_HE_GFAP_conch,
    Simple_MLP_HE_KI67_GFAP_conch
    
    ])

# %% SAVE CSV
path = '/local/data1/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results'
summary.to_csv(f'{path}/5_class_summary.csv', index=False)

# %%