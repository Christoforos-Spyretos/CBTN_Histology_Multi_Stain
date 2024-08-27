# %% IMPORTS
import os
import pandas as pd

# %% LOAD THE NON-MERGED SPLITS
path = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/splits/HE_7_class_tumor_subtyping_100"
save_path = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/splits/Merged_HE_7_class_tumor_subtyping_100"

if not os.path.exists(save_path):
        os.makedirs(save_path)

# %%
for i in range(5):
    split_bool_file_path = os.path.join(path, f'splits_{i}_bool.csv')
    splits_bool = pd.read_csv(split_bool_file_path)
    splits_bool['slide_id'] = splits_bool['slide_id'].apply(lambda x: x.split('___')[0])
    splits_bool = splits_bool.drop_duplicates(subset='slide_id', keep='last')
    split_save_path = os.path.join(save_path, f'splits_{i}_bool.csv')
    splits_bool.to_csv(split_save_path, index=False)

# %%
for i in range(5):
    splits_file_path = os.path.join(path, f'splits_{i}.csv')
    splits = pd.read_csv(splits_file_path)

    new_splits = pd.DataFrame()

    train = []

    for subject_session_wsi in splits["train"]:
        subject_id = subject_session_wsi.split('___')[0]
        train.append(subject_id)

    new_splits['train'] = train
    new_splits['train'] = new_splits['train'].drop_duplicates()
    new_splits.dropna(subset=['train'], inplace = True)

    val_df = pd.DataFrame()
    val = []

    for subject_session_wsi in splits["val"]:
        if not pd.isnull(subject_session_wsi):
            subject_id = subject_session_wsi.split('___')[0]
            val.append(subject_id)

    val_df['val'] = val
    val_df['val'] = val_df['val'].drop_duplicates()
    val_df.dropna(subset=['val'], inplace = True)

    val_df = val_df.join(new_splits['train'], how='right')
    new_splits = new_splits.join(val_df['val'], how='right')
    new_splits = new_splits.reset_index()
    new_splits.insert(loc=1, column='slide_id', value=new_splits.index)
    new_splits = new_splits.drop('index', axis=1)

    test_df = pd.DataFrame()
    test = []
 
    for subject_session_wsi in splits["test"]:
        if not pd.isnull(subject_session_wsi):
            subject_id = subject_session_wsi.split('___')[0]
            test.append(subject_id)

    test_df['test'] = test
    test_df['test'] = test_df['test'].drop_duplicates()
    test_df.dropna(subset=['test'], inplace = True)

    test_df = test_df.join(new_splits['val'], how='right')
    new_splits = new_splits.join(test_df['test'], how='right')        

    splits_save_path = os.path.join(save_path, f'splits_{i}.csv')
    new_splits.to_csv(splits_save_path, index=False)

# %%
histology_df = pd.read_csv("/local/data2/chrsp39/CBTN_v2/CBTN_histology_summary.csv")

histology_df = histology_df[['subjectID','diagnosis']]

tumour_types = [
    'Low-grade glioma/astrocytoma (WHO grade I/II)',
    'High-grade glioma/astrocytoma (WHO grade III/IV)', 
    'Medulloblastoma',
    'Ependymoma',
    'Brainstem glioma- Diffuse intrinsic pontine glioma',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)',
    'Meningioma',
    'Craniopharyngioma',
    'Dysembryoplastic neuroepithelial tumor (DNET)',
    'Ganglioglioma'

    ]
 
histology_df = histology_df[histology_df['diagnosis'].isin(tumour_types)]

histology_df['diagnosis'] = histology_df['diagnosis'].replace({
    'Low-grade glioma/astrocytoma (WHO grade I/II)': 'ASTR_LGG',
    'High-grade glioma/astrocytoma (WHO grade III/IV)': 'ASTR_HGG', 
    'Medulloblastoma': 'MED',
    'Ependymoma': 'EP',
    'Brainstem glioma- Diffuse intrinsic pontine glioma': 'DIPG',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)': 'ATRT',
    'Meningioma': 'MEN',
    'Craniopharyngioma': 'CRAN',
    'Dysembryoplastic neuroepithelial tumor (DNET)': 'DNET',
    'Ganglioglioma': 'GANG'
})

histology_df = histology_df.drop_duplicates(subset='subjectID', keep='last')

diagnoses = ["ASTR_LGG", "ASTR_HGG", "MED", "EP", "ATRT", "MEN", "GANG"]

for i in range(5):

    splits_descriptor_data = []

    splits_file_path = os.path.join(save_path, f'splits_{i}.csv')
    splits = pd.read_csv(splits_file_path)

    splits['train'] = splits['train'].fillna('')
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

# %% CONCAT CSV FILES FOR ALL HE 7 OR 10 CLASSES
path_1 = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/splits/HE_3_class_tumor_subtyping_100"
path_2 = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/splits/HE_7_class_tumor_subtyping_100"
save_path = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/splits/HE_10_class_tumor_subtyping_100"

# %%
for i in range(5):
    split_bool_file_path_1 = os.path.join(path_1, f'splits_{i}_bool.csv')
    split_bool_file_path_2 = os.path.join(path_2, f'splits_{i}_bool.csv')
    splits_bool_1 = pd.read_csv(split_bool_file_path_1)
    splits_bool_2 = pd.read_csv(split_bool_file_path_2)
    splits_bool = pd.concat([splits_bool_1, splits_bool_2])
    split_save_path = os.path.join(save_path, f'splits_{i}_bool.csv')
    splits_bool.to_csv(split_save_path, index=False)

# %% 
for i in range(5):
    split_file_path_1 = os.path.join(path_1, f'splits_{i}.csv')
    split_file_path_2 = os.path.join(path_2, f'splits_{i}.csv')
    splits_1 = pd.read_csv(split_file_path_1)
    splits_2 = pd.read_csv(split_file_path_2)
    splits = pd.concat([splits_1, splits_2])
    split_save_path = os.path.join(save_path, f'splits_{i}.csv')
    splits.to_csv(split_save_path, index=False)
    
# %%
histology_df = pd.read_csv("/local/data2/chrsp39/CBTN_v2/CBTN_histology_summary.csv")

histology_df = histology_df[['subjectID','diagnosis']]

tumour_types = [
    'Low-grade glioma/astrocytoma (WHO grade I/II)',
    'High-grade glioma/astrocytoma (WHO grade III/IV)', 
    'Medulloblastoma',
    'Ependymoma',
    'Brainstem glioma- Diffuse intrinsic pontine glioma',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)',
    'Meningioma',
    'Craniopharyngioma',
    'Dysembryoplastic neuroepithelial tumor (DNET)',
    'Ganglioglioma'

    ]
 
histology_df = histology_df[histology_df['diagnosis'].isin(tumour_types)]

histology_df['diagnosis'] = histology_df['diagnosis'].replace({
    'Low-grade glioma/astrocytoma (WHO grade I/II)': 'ASTR_LGG',
    'High-grade glioma/astrocytoma (WHO grade III/IV)': 'ASTR_HGG', 
    'Medulloblastoma': 'MED',
    'Ependymoma': 'EP',
    'Brainstem glioma- Diffuse intrinsic pontine glioma': 'DIPG',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)': 'ATRT',
    'Meningioma': 'MEN',
    'Craniopharyngioma': 'CRAN',
    'Dysembryoplastic neuroepithelial tumor (DNET)': 'DNET',
    'Ganglioglioma': 'GANG'
})

histology_df = histology_df.drop_duplicates(subset='subjectID', keep='last')

diagnoses = ["ASTR_LGG", "ASTR_HGG", "MED", "EP", "ATRT", "MEN", "GANG", "DIPG", "DNET", "CRAN"]

for i in range(5):

    splits_descriptor_data = []

    splits_file_path = os.path.join(save_path, f'splits_{i}.csv')
    splits = pd.read_csv(splits_file_path)

    splits['train'] = splits['train'].fillna('')
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
