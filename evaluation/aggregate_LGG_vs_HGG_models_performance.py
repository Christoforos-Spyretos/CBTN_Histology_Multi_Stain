# %% IMPORTS
import pandas as pd

# %% LOAD CSV

HE_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_small_clam_sb_conch_v1.csv')
KI67_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_KI67_small_clam_sb_conch_v1.csv')
EF_HE_KI67_conch = pd.read_csv('/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_LGG_vs_HGG_Merged_HE_KI67_small_clam_sb_conch_v1.csv')

# %% AGGREGATE CSV FILES
# Aggregate all models
summary = pd.concat([
    HE_conch, 
    KI67_conch,
    
    EF_HE_KI67_conch,
    
    # AP_HE_KI67_conch,
    # AP_HE_GFAP_conch,
    # AP_HE_KI67_GFAP_conch,
    
    # AG_HE_KI67_conch,
    # AG_HE_GFAP_conch,
    # AG_HE_KI67_GFAP_conch,

    # MAJ_VOT_HE_KI67_conch,
    # MAJ_VOT_HE_GFAP_conch,
    # MAJ_VOT_HE_KI67_GFAP_conch,

    # Simple_MLP_HE_KI67_conch,
    # Simple_MLP_HE_GFAP_conch,
    # Simple_MLP_HE_KI67_GFAP_conch
    
    ])

# %% SAVE CSV
path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results'
summary.to_csv(f'{path}/LGG_vs_HGG_summary.csv', index=False)

# %%