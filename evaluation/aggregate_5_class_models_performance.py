# %% IMPORTS
import pandas as pd

# %% LOAD CSV

HE_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_HE_small_clam_sb_conch_v1.csv')
HE_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_HE_small_clam_sb_conch_v1_5.csv')
# HE_resnet50_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_HE_small_clam_sb_resnet50.csv')
HE_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_HE_small_clam_sb_uni.csv')
HE_uni2_h_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_HE_small_clam_sb_uni2-h.csv')


KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_KI67_small_clam_sb_conch_v1.csv')
KI67_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_KI67_small_clam_sb_conch_v1_5.csv')
# KI67_resnet50_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_KI67_small_clam_sb_resnet50.csv')
KI67_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_KI67_small_clam_sb_uni.csv')
KI67_uni2_h_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_KI67_small_clam_sb_uni2-h.csv')

EF_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Early_Fusion_HE_KI67_small_clam_sb_conch_v1.csv')
EF_HE_KI67_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Early_Fusion_HE_KI67_small_clam_sb_conch_v1_5.csv')
# EF_HE_KI67_resnet50_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Early_Fusion_HE_KI67_small_clam_sb_resnet50.csv')
EF_HE_KI67_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Early_Fusion_HE_KI67_small_clam_sb_uni.csv')
EF_HE_KI67_uni2_h_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Early_Fusion_HE_KI67_small_clam_sb_uni2-h.csv')

IM_CA_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_conch_v1.csv')
IM_CA_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_conch_v1_5.csv')   
# IM_CA_HE_KI67_resnet50 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_resnet50.csv')   
IM_CA_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_uni.csv')   
IM_CA_HE_KI67_uni2_h = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_uni2-h.csv')   

IM_EWM_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_EWM_HE_KI67_small_clam_sb_conch_v1_5.csv')
IM_CONCAT_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Intermediate_Fusion_CONCAT_HE_KI67_small_clam_sb_conch_v1_5.csv')

LF_PA_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1.csv')
LF_LA_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1.csv')
LF_SM_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_OHL_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_THL_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_AM_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1.csv')

LF_PA_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1_5.csv')   
LF_LA_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_SM_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_OHL_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_THL_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_AM_HE_KI67_conch_v1_5 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1_5.csv')

LF_PA_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_PA_HE_KI67_small_clam_sb_uni.csv')   
LF_LA_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LA_HE_KI67_small_clam_sb_uni.csv')
LF_SM_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_uni.csv')
LF_OHL_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_uni.csv')
LF_THL_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_uni.csv')
LF_AM_HE_KI67_uni = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_uni.csv')

LF_PA_HE_KI67_uni2_h = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_PA_HE_KI67_small_clam_sb_uni2-h.csv')   
LF_LA_HE_KI67_uni2_h = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LA_HE_KI67_small_clam_sb_uni2-h.csv')
LF_SM_HE_KI67_uni2_h = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_uni2-h.csv')
LF_OHL_HE_KI67_uni2_h = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_uni2-h.csv')
LF_THL_HE_KI67_uni2_h = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_uni2-h.csv')
LF_AM_HE_KI67_uni2_h = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class/EVAL_5_class_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_uni2-h.csv')

# %% AGGREGATE CSV FILES
# Aggregate all models
summary = pd.concat([
    HE_conch_v1_small_clam_sb,
    HE_conch_v1_5_small_clam_sb,
    # HE_resnet50_small_clam_sb,
    HE_uni_small_clam_sb,
    HE_uni2_h_small_clam_sb,

    KI67_conch_v1_small_clam_sb,
    KI67_conch_v1_5_small_clam_sb,
    # KI67_resnet50_small_clam_sb,
    KI67_uni_small_clam_sb,
    KI67_uni2_h_small_clam_sb,

    EF_HE_KI67_conch_v1_small_clam_sb,
    EF_HE_KI67_conch_v1_5_small_clam_sb,
    # EF_HE_KI67_resnet50_small_clam_sb,
    EF_HE_KI67_uni_small_clam_sb,
    EF_HE_KI67_uni2_h_small_clam_sb,

    IM_CA_HE_KI67_conch_v1,
    IM_CA_HE_KI67_conch_v1_5,
    IM_CA_HE_KI67_uni,
    IM_CA_HE_KI67_uni2_h,

    IM_EWM_HE_KI67_conch_v1_5,
    IM_CONCAT_HE_KI67_conch_v1_5,

    LF_PA_HE_KI67_conch_v1,
    LF_LA_HE_KI67_conch_v1,
    LF_SM_HE_KI67_conch_v1,
    LF_OHL_HE_KI67_conch_v1,
    LF_THL_HE_KI67_conch_v1,
    LF_AM_HE_KI67_conch_v1,

    LF_PA_HE_KI67_conch_v1_5,
    LF_LA_HE_KI67_conch_v1_5,
    LF_SM_HE_KI67_conch_v1_5,
    LF_OHL_HE_KI67_conch_v1_5,
    LF_THL_HE_KI67_conch_v1_5,
    LF_AM_HE_KI67_conch_v1_5,

    LF_PA_HE_KI67_uni,
    LF_LA_HE_KI67_uni,
    LF_SM_HE_KI67_uni,
    LF_OHL_HE_KI67_uni,
    LF_THL_HE_KI67_uni,
    LF_AM_HE_KI67_uni,

    LF_PA_HE_KI67_uni2_h,
    LF_LA_HE_KI67_uni2_h,
    LF_SM_HE_KI67_uni2_h,
    LF_OHL_HE_KI67_uni2_h,
    LF_THL_HE_KI67_uni2_h,
    LF_AM_HE_KI67_uni2_h,
])

# %% SAVE CSV
path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/5_class'
summary.to_csv(f'{path}/5_class_summary.csv', index=False)

# %%