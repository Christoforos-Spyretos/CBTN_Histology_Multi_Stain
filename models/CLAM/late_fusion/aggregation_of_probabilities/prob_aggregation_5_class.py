# %% IMPORTS
import pandas as pd
import os

# %% LOAD RESULTS
# path to results
Merged_HE_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_5_class_Merged_HE_small_clam_sb_conch_v1'
Merged_KI67_path = '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_5_class_Merged_KI67_small_clam_sb_conch_v1'

HE_contents = os.listdir(Merged_HE_path)
HE_folds_dict = {} 

for content in HE_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_HE_path + '/' + content)
        HE_folds_dict[name] = df

KI67_contents = os.listdir(Merged_KI67_path)
KI67_folds_dict = {}

for content in KI67_contents:
    if content.endswith(".csv"):
        name = content.split('.')[0]
        df = pd.read_csv(Merged_KI67_path + '/' + content)
        KI67_folds_dict[name] = df  

folds = [f'fold_{i}' for i in range(50)]

# %% MERGED_HE_KI67
for fold in folds:
    HE_fold = HE_folds_dict[fold]
    KI67_fold = KI67_folds_dict[fold]
    slide_id = HE_fold['slide_id']
    p_0 = (HE_fold['p_0'] + KI67_fold['p_0']) / 2
    p_1 = (HE_fold['p_1'] + KI67_fold['p_1']) / 2
    p_2 = (HE_fold['p_2'] + KI67_fold['p_2']) / 2
    p_3 = (HE_fold['p_3'] + KI67_fold['p_3']) / 2
    p_4 = (HE_fold['p_4'] + KI67_fold['p_4']) / 2
    Y = HE_fold['Y']
    Y_hat = [None] * len(Y)
    for i in range(len(p_0)):
        # get the max probability
        max_prob = max(p_0[i], p_1[i], p_2[i], p_3[i], p_4[i])
        # assign the class label based
        if max_prob == p_0[i]:
            Y_hat[i] = 0
        elif max_prob == p_1[i]:
            Y_hat[i] = 1
        elif max_prob == p_2[i]:
            Y_hat[i] = 2
        elif max_prob == p_3[i]:
            Y_hat[i] = 3
        else:
            Y_hat[i] = 4
    Merged_HE_KI67_fold = pd.DataFrame({
        'slide_id': slide_id, 
        'Y': Y, 
        'Y_hat': Y_hat, 
        'p_0': p_0, 
        'p_1': p_1,
        'p_2': p_2,
        'p_3': p_3,
        'p_4': p_4})
    # Define save path
    save_path = f'/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/CLAM/eval_results/EVAL_5_class_Late_Fusion_PA_HE_KI67_small_clam_sb_conch_v1/{fold}.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save results
    Merged_HE_KI67_fold.to_csv(save_path, index=False)
# %%


