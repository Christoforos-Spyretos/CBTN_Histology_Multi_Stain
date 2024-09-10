# %% IMPORTS
import os
import pandas as pd
import numpy as np

# %%
KI67_splits_path = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/splits/KI67_7_class_tumor_subtyping_100"
save_path = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/splits/HE_7_class_tumor_subtyping_100"
HE_df = pd.read_csv("/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_7_class_dataset.csv")

# %%
for i in range(5):

    split_bool_file_path = os.path.join(KI67_splits_path, f'splits_0_bool.csv')
    splits_bool = pd.read_csv(split_bool_file_path)

    for slide_id in splits_bool['Unnamed: 0']:
        new_slide_id = slide_id.split('___')[0]
        splits_bool.loc[splits_bool['Unnamed: 0'] == slide_id, 'Unnamed: 0'] = new_slide_id

    splits_bool = splits_bool[splits_bool['Unnamed: 0'].isin(HE_df['case_id'])]

    splits_bool = splits_bool.drop_duplicates(subset=['Unnamed: 0'], keep='first')

    new_splits_bool = pd.DataFrame(columns=splits_bool.columns)

    for slide_id in HE_df['slide_id']:
        case_id = slide_id.split('___')[0]
        if case_id in splits_bool['Unnamed: 0'].values:
            matched_row = splits_bool[splits_bool['Unnamed: 0'] == case_id]
            matched_row['Unnamed: 0'] = slide_id
            new_splits_bool = pd.concat([new_splits_bool, matched_row])

    split_save_path = os.path.join(save_path, f'splits_{i}_bool.csv')
    new_splits_bool.to_csv(split_save_path, index=False)

# %%
for i in range (5):

    split_bool_file_path = os.path.join(save_path, f'splits_0_bool.csv')
    splits_bool = pd.read_csv(split_bool_file_path)

    train_subjects = splits_bool[splits_bool['train']]['Unnamed: 0'].tolist()
    val_subjects = splits_bool[splits_bool['val']]['Unnamed: 0'].tolist()
    test_subjects = splits_bool[splits_bool['test']]['Unnamed: 0'].tolist()

    max_length = max(len(train_subjects), len(val_subjects), len(test_subjects))

    new_splits = pd.DataFrame(index=range(max_length))

    new_splits['Unnamed: 0'] = range(max_length)
    new_splits['train'] = train_subjects + [np.nan] * (max_length - len(train_subjects))
    new_splits['val'] = val_subjects + [np.nan] * (max_length - len(val_subjects))
    new_splits['test'] = test_subjects + [np.nan] * (max_length - len(test_subjects))

    split_save_path = os.path.join(save_path, f'splits_{i}.csv')
    new_splits.to_csv(split_save_path, index=False)

# %%
histology_df = pd.read_csv("/local/data2/chrsp39/CBTN_v2/CBTN_histology_summary.csv")

histology_df = histology_df[['subjectID','diagnosis']]

tumour_types = [
    'Low-grade glioma/astrocytoma (WHO grade I/II)',
    'High-grade glioma/astrocytoma (WHO grade III/IV)', 
    'Medulloblastoma',
    'Ependymoma',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)',
    'Meningioma',
    'Ganglioglioma'
    ]
 
histology_df = histology_df[histology_df['diagnosis'].isin(tumour_types)]

histology_df['diagnosis'] = histology_df['diagnosis'].replace({
    'Low-grade glioma/astrocytoma (WHO grade I/II)': 'ASTR_LGG',
    'High-grade glioma/astrocytoma (WHO grade III/IV)': 'ASTR_HGG', 
    'Medulloblastoma': 'MED',
    'Ependymoma': 'EP',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)': 'ATRT',
    'Meningioma': 'MEN',
    'Ganglioglioma': 'GANG'
})

histology_df = histology_df.drop_duplicates(subset='subjectID', keep='last')

diagnoses = ["ASTR_LGG", "ASTR_HGG", "MED", "EP", "ATRT", "MEN", "GANG"]

for i in range(5):

    splits_descriptor_data = []

    splits_file_path = os.path.join(save_path, f'splits_{i}.csv')
    splits = pd.read_csv(splits_file_path)

    splits['val'] = splits['val'].fillna('')
    splits['test'] = splits['test'].fillna('')

    splits_descriptor = pd.DataFrame(columns=['diagnosis', 'train', 'val', 'test'])

    for diagnosis in diagnoses:
        train_subjects = histology_df[histology_df['subjectID'].isin(splits['train'].apply(lambda x: x.split('___')[0])) & (histology_df['diagnosis'] == diagnosis)].shape[0]    
        val_subjects = histology_df[histology_df['subjectID'].isin(splits['val'].apply(lambda x: x.split('___')[0])) & (histology_df['diagnosis'] == diagnosis)].shape[0]    
        test_subjects = histology_df[histology_df['subjectID'].isin(splits['test'].apply(lambda x: x.split('___')[0])) & (histology_df['diagnosis'] == diagnosis)].shape[0]    
        splits_descriptor_data.append({'diagnosis': diagnosis, 'train': train_subjects, 'val': val_subjects, 'test': test_subjects})

        splits_descriptor = pd.DataFrame(splits_descriptor_data)
        splits_descriptor_save_path = os.path.join(save_path, f'splits_{i}_descriptor.csv')
        splits_descriptor.to_csv(splits_descriptor_save_path, index=False)        

# %%