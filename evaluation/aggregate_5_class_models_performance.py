# %% IMPORTS
import pandas as pd

# %% LOAD CSV
HE_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_HE_small_clam_sb_conch_v1_5.csv')
KI67_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_KI67_small_clam_sb_conch_v1_5.csv')

EF_HE_KI67_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Early_Fusion_HE_KI67_small_clam_sb_conch_v1_5.csv')

IM_CA_HE_inform_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Intermediate_Fusion_CA_HE_inform_KI67_small_clam_sb_conch_v1_5.csv')   
IM_CA_KI67_inform_HE_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Intermediate_Fusion_CA_KI67_inform_HE_small_clam_sb_conch_v1_5.csv')
IM_EWM_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Intermediate_Fusion_EWM_HE_KI67_small_clam_sb_conch_v1_5.csv')
IM_CONCAT_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Intermediate_Fusion_CONCAT_HE_KI67_small_clam_sb_conch_v1_5.csv')

LF_PA_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1_5.csv')   
LF_LA_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_SM_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_OHL_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_THL_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_AM_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1_5.csv')

# %% AGGREGATE CSV FILES
# Aggregate all models
summary = pd.concat([
    HE_conch_v1_5_small_clam_sb,
  
    KI67_conch_v1_5_small_clam_sb,
    
    EF_HE_KI67_conch_v1_5_small_clam_sb,
   
    IM_CA_HE_inform_KI67_conch_v1_5,
    IM_CA_KI67_inform_HE_conch_v1_5,
    IM_EWM_HE_KI67_conch_v1_5,
    IM_CONCAT_HE_KI67_conch_v1_5,

    LF_PA_HE_KI67_conch_v1_5,
    LF_LA_HE_KI67_conch_v1_5,
    LF_SM_HE_KI67_conch_v1_5,
    LF_OHL_HE_KI67_conch_v1_5,
    LF_THL_HE_KI67_conch_v1_5,
    LF_AM_HE_KI67_conch_v1_5,
])

# %% SAVE CSV
path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/70%_split/5_class'
summary.to_csv(f'{path}/5_class_summary.csv', index=False)

# %%