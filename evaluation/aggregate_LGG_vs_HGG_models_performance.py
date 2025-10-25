# %% IMPORTS
import pandas as pd

# %% LOAD CSV

HE_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1.csv')
KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1.csv')

EF_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_conch_v1.csv')

IM_CA_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_conch_v1.csv')

LF_PA_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1.csv')
LF_LA_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1.csv')
LF_MJ_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_MJ_HE_KI67_small_clam_sb_conch_v1.csv')
LF_SM_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_OHL_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_THL_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_AM_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1.csv')

# %% AGGREGATE CSV FILES
# Aggregate all models
summary = pd.concat([
    HE_conch_v1, 
    KI67_conch_v1,
    
    EF_HE_KI67_conch_v1,

    IM_CA_HE_KI67_conch_v1,

    LF_PA_HE_KI67_conch_v1,
    LF_LA_HE_KI67_conch_v1,
    LF_MJ_HE_KI67_conch_v1,
    LF_SM_HE_KI67_conch_v1,
    LF_OHL_HE_KI67_conch_v1,
    LF_AM_HE_KI67_conch_v1
])

# %% SAVE CSV
path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG'
summary.to_csv(f'{path}/LGG_vs_HGG_summary.csv', index=False)

# %%