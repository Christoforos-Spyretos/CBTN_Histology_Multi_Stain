# %% IMPORTS
import pandas as pd

# %% LOAD CSV

HE_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1.csv')
HE_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1_5.csv')
# HE_resnet50_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_resnet50.csv')
HE_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_uni.csv')
# HE_uni2_h_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_uni2-h.csv')

# HE_conch_v1_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE___abmil_conch_v1.csv')
# HE_conch_v1_5_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE___abmil_conch_v1_5.csv')
# HE_resnet50_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE___abmil_resnet50.csv')
# HE_uni_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE___abmil_uni.csv')
# HE_uni2_h_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE___abmil_uni2-h.csv')

KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1.csv')
KI67_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1_5.csv')
# KI67_resnet50_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_resnet50.csv')
KI67_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_uni.csv')
# KI67_uni2_h_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_uni2-h.csv')

# KI67_conch_v1_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67___abmil_conch_v1.csv')
# KI67_conch_v1_5_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67___abmil_conch_v1_5.csv')
# KI67_resnet50_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67___abmil_resnet50.csv')
# KI67_uni_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67___abmil_uni.csv')
# KI67_uni2_h_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67___abmil_uni2-h.csv')

EF_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_conch_v1.csv')
EF_HE_KI67_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_conch_v1_5.csv')
# EF_HE_KI67_resnet50_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_resnet50.csv')
EF_HE_KI67_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_uni.csv')
# EF_HE_KI67_uni2_h_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67_small_clam_sb_uni2-h.csv')

# EF_HE_only_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only_small_clam_sb_conch_v1.csv')
# EF_HE_only_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only_small_clam_sb_conch_v1_5.csv')
# EF_HE_only_resnet50_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only_small_clam_sb_resnet50.csv')
# EF_HE_only_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only_small_clam_sb_uni.csv')
# EF_HE_only_uni2_h_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only_small_clam_sb_uni2-h.csv')

# EF_KI67_only_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only_small_clam_sb_conch_v1.csv')
# EF_KI67_only_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only_small_clam_sb_conch_v1_5.csv')
# EF_KI67_only_resnet50_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only_small_clam_sb_resnet50.csv')    
# EF_KI67_only_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only_small_clam_sb_uni.csv')
# EF_KI67_only_uni2_h_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only_small_clam_sb_uni2-h.csv')

# EF_HE_KI67_conch_v1_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67___abmil_conch_v1.csv')
# EF_HE_KI67_conch_v1_5_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67___abmil_conch_v1_5.csv')
# EF_HE_KI67_resnet50_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67___abmil_resnet50.csv')
# EF_HE_KI67_uni_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67___abmil_uni.csv')
# EF_HE_KI67_uni2_h_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_KI67___abmil_uni2-h.csv')

# EF_HE_only_conch_v1_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only___abmil_conch_v1.csv')
# EF_HE_only_conch_v1_5_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only___abmil_conch_v1_5.csv')
# EF_HE_only_resnet50_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only___abmil_resnet50.csv')
# EF_HE_only_uni_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only___abmil_uni.csv')
# EF_HE_only_uni2_h_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_HE_only___abmil_uni2-h.csv')

# EF_KI67_only_conch_v1_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only___abmil_conch_v1.csv')
# EF_KI67_only_conch_v1_5_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only___abmil_conch_v1_5.csv')
# EF_KI67_only_resnet50_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only___abmil_resnet50.csv')    
# EF_KI67_only_uni_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only___abmil_uni.csv')
# EF_KI67_only_uni2_h_abmil = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Early_Fusion_KI67_only___abmil_uni2-h.csv')

IM_CA_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Intermediate_Fusion_CA_HE_KI67_small_clam_sb_conch_v1.csv')

LF_PA_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1.csv')
LF_LA_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1.csv')
# LF_MJ_HE_KI67_conch_v1 = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_MJ_HE_KI67_small_clam_sb_conch_v1.csv')
LF_SM_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_SL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_OHL_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_OHL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_THL_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_THL_HE_KI67_small_clam_sb_conch_v1.csv')
LF_AM_HE_KI67_conch_v1_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LM_AL_HE_KI67_small_clam_sb_conch_v1.csv')

LF_PA_HE_KI67_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1_5.csv')
LF_LA_HE_KI67_conch_v1_5_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_conch_v1_5.csv')

LF_PA_HE_KI67_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_uni.csv')
LF_LA_HE_KI67_uni_small_clam_sb = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_LA_HE_KI67_small_clam_sb_uni.csv')
# %% AGGREGATE CSV FILES
# Aggregate all models
summary = pd.concat([
    HE_conch_v1_small_clam_sb,
    HE_conch_v1_5_small_clam_sb,
    # HE_resnet50_small_clam_sb,
    HE_uni_small_clam_sb,
    # HE_uni2_h_small_clam_sb,

    # HE_conch_v1_abmil,
    # HE_conch_v1_5_abmil,
    # HE_resnet50_abmil,
    # HE_uni_abmil,
    # HE_uni2_h_abmil,

    KI67_conch_v1_small_clam_sb,
    KI67_conch_v1_5_small_clam_sb,
    # KI67_resnet50_small_clam_sb,
    KI67_uni_small_clam_sb,
    # KI67_uni2_h_small_clam_sb,

    # KI67_conch_v1_abmil,
    # KI67_conch_v1_5_abmil,
    # KI67_resnet50_abmil,
    # KI67_uni_abmil,
    # KI67_uni2_h_abmil,

    EF_HE_KI67_conch_v1_small_clam_sb,
    EF_HE_KI67_conch_v1_5_small_clam_sb,
    # EF_HE_KI67_resnet50_small_clam_sb,
    EF_HE_KI67_uni_small_clam_sb,
    # EF_HE_KI67_uni2_h_small_clam_sb,

    # EF_HE_only_conch_v1_small_clam_sb,
    # EF_HE_only_conch_v1_5_small_clam_sb,
    # EF_HE_only_resnet50_small_clam_sb,
    # EF_HE_only_uni_small_clam_sb,
    # EF_HE_only_uni2_h_small_clam_sb,

    # EF_KI67_only_conch_v1_small_clam_sb,
    # EF_KI67_only_conch_v1_5_small_clam_sb,
    # EF_KI67_only_resnet50_small_clam_sb,    
    # EF_KI67_only_uni_small_clam_sb,
    # EF_KI67_only_uni2_h_small_clam_sb,  

    # EF_HE_KI67_conch_v1_abmil,
    # EF_HE_KI67_conch_v1_5_abmil,
    # EF_HE_KI67_resnet50_abmil,
    # EF_HE_KI67_uni_abmil,
    # EF_HE_KI67_uni2_h_abmil,

    # EF_HE_only_conch_v1_abmil,
    # EF_HE_only_conch_v1_5_abmil,
    # EF_HE_only_resnet50_abmil,      
    # EF_HE_only_uni_abmil,
    # EF_HE_only_uni2_h_abmil,

    # EF_KI67_only_conch_v1_abmil,
    # EF_KI67_only_conch_v1_5_abmil,
    # EF_KI67_only_resnet50_abmil,    
    # EF_KI67_only_uni_abmil,
    # EF_KI67_only_uni2_h_abmil,
    
    IM_CA_HE_KI67_conch_v1_small_clam_sb,

    LF_PA_HE_KI67_conch_v1_small_clam_sb,
    LF_LA_HE_KI67_conch_v1_small_clam_sb,
    # LF_MJ_HE_KI67_conch_v1,
    LF_SM_HE_KI67_conch_v1_small_clam_sb,
    LF_OHL_HE_KI67_conch_v1_small_clam_sb,
    LF_THL_HE_KI67_conch_v1_small_clam_sb,
    LF_AM_HE_KI67_conch_v1_small_clam_sb,

    LF_PA_HE_KI67_conch_v1_5_small_clam_sb,
    LF_LA_HE_KI67_conch_v1_5_small_clam_sb,

    LF_PA_HE_KI67_uni_small_clam_sb,
    LF_LA_HE_KI67_uni_small_clam_sb,
])

# %% SAVE CSV
path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/evaluation/LGG_vs_HGG'
summary.to_csv(f'{path}/LGG_vs_HGG_summary.csv', index=False)

# %%