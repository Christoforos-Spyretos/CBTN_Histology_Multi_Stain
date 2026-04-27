# %% IMPORTS
import pandas as pd
import os

# %% LOAD RESULTS
# path to results
HE_results_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split_0.5_training_drop/LGG_vs_HGG/EVAL_LGG_vs_HGG_HE_small_clam_sb_conch_v1_5'
KI67_results_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split_0.5_training_drop/LGG_vs_HGG/EVAL_LGG_vs_HGG_KI67_small_clam_sb_conch_v1_5'

HE_contents = os.listdir(HE_results_path)
HE_folds_dict = {} 

for content in HE_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(HE_results_path + '/' + content)
        HE_folds_dict[name] = df

KI67_contents = os.listdir(KI67_results_path)
KI67_folds_dict = {}

for content in KI67_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(KI67_results_path + '/' + content)
        KI67_folds_dict[name] = df  

folds = [f'fold_{i}' for i in range(50)]

# %% MERGED_HE_KI67
for fold in folds:
    HE_fold = HE_folds_dict[fold]
    KI67_fold = KI67_folds_dict[fold]
    slide_id = HE_fold['slide_id']
    p_0 = (HE_fold['p_0'] + KI67_fold['p_0']) / 2
    p_1 = (HE_fold['p_1'] + KI67_fold['p_1']) / 2
    Y = HE_fold['Y']
    Y_hat = [None] * len(Y)
    for i in range(len(p_0)):
        if p_0[i] > p_1[i]:
            Y_hat[i] = 0
        else:
            Y_hat[i] = 1
    Merged_HE_KI67_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat, 
        'p_0': p_0, 
        'p_1': p_1})
    # define save path
    save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/50%_split_0.5_training_drop/LGG_vs_HGG/EVAL_LGG_vs_HGG_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1_5/{fold}.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # save results
    Merged_HE_KI67_fold.to_csv(save_path, index=False)

# %%


