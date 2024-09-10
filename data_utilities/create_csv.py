# %% IMPORTS
import pandas as pd
import os

# %% LOAD DATA
histology_images = "/local/data2/chrsp39/CBTN_v2/GFAP/WSI"
histology_df = pd.read_csv("/local/data2/chrsp39/CBTN_v2/CBTN_histology_summary.csv")

# %% CLEAN HISTOLOGY DATAFRAME
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

# %% CREATE CSV FOR 10 CLASSES
df_csv_list = []
subjects = os.listdir(histology_images)

for subject in subjects:
    file_name = os.path.splitext(subject)[0] 
    case_id = file_name.split("___")[0]
    slide_id = "___".join(file_name.split("___")[0:])
    if case_id in histology_df['subjectID'].values:
        label = histology_df.loc[histology_df['subjectID'] == case_id, 'diagnosis'].values[0]
        df_csv_list.append({'case_id': case_id, 'slide_id': slide_id, 'label': label})

dataset_csv = pd.DataFrame(df_csv_list)

dataset_csv.to_csv('/local/data3/chrsp39/CBTN_v2/CSV_FILES/GFAP/GFAP_10_class_dataset.csv', index=False)

# %% CREATE CSV FOR SPECIFIC CLASSES
class_10_csv = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSV_FILES/GFAP/GFAP_10_class_dataset.csv")
class_2_csv = class_10_csv.drop(class_10_csv[
    (class_10_csv['label'] == 'DNET') | 
    (class_10_csv['label'] == 'DIPG') | 
    (class_10_csv['label'] == 'CRAN') |
    (class_10_csv['label'] == 'ATRT') |
    (class_10_csv['label'] == 'MEN')  |  
    (class_10_csv['label'] == 'EP')   |  
    (class_10_csv['label'] == 'GANG') |
    (class_10_csv['label'] == 'MED') 
    ].index)
class_2_csv.to_csv('/local/data3/chrsp39/CBTN_v2/CSV_FILES/GFAP/GFAP_HGG_LGG_dataset.csv', index=False)

# %% CREATE BOUNDED CSV 
df_HE = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSV_FILES/HE/HE_HGG_LGG_dataset.csv")
df_KI67 = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSV_FILES/KI67/KI67_HGG_LGG_dataset.csv")
df_GFAP = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSV_FILES/GFAP/GFAP_HGG_LGG_dataset.csv")

HE_subjects = set(df_HE['case_id'].values)
KI67_subjects = set(df_KI67['case_id'].values)
GFAP_subjects = set(df_GFAP['case_id'].values)

common_subjects = HE_subjects & KI67_subjects & GFAP_subjects

df_HE_common = df_HE[df_HE['case_id'].isin(common_subjects)]
df_HE_common.to_csv('/local/data3/chrsp39/CBTN_v2/CSV_FILES/Merged_HE_KI67_GFAP/Merged_HE_KI67_GFAP_HGG_LGG_dataset.csv', index=False)

# %% CREATE MERGED CSV
df = pd.read_csv('/local/data2/chrsp39/CBTN_v2/UNI/GFAP/GFAP_HGG_LGG_dataset_bounded.csv')

df['slide_id'] = df['slide_id'].apply(lambda x: x.split('___')[0])

merged_df = df.drop_duplicates(subset=['case_id'], keep='last')
merged_df.to_csv('/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67_GFAP/Merged_HE_KI67_GFAP_HGG_LGG_dataset_bounded.csv', index=False)

# %% MERGED HISTOLOGY CSV
df = pd.read_csv('/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE/Merged_HE_5_class_dataset_bounded.csv')
df.to_csv('/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67_GFAP/Merged_HE_KI67_GFAP_5_class_dataset_bounded.csv', index=False)

# %% CREATE 3-CLASS CSV
HE_10_class_dataset = pd.read_csv('/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_10_class_dataset.csv')
HE_7_class_dataset = pd.read_csv('/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_7_class_dataset.csv')

HE_case_ids = set(HE_10_class_dataset['case_id'])
HE_7_class_case_ids = set(HE_7_class_dataset['case_id'])

ids_to_keep = HE_case_ids - HE_7_class_case_ids

HE_3_class_remains = HE_10_class_dataset[HE_10_class_dataset['case_id'].isin(ids_to_keep)]

HE_3_class_remains.to_csv('/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_3_class_dataset.csv', index=False)

# dataset = pd.read_csv('/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_3_class_dataset.csv')

# %% CREATE 10-CLASS MERGED HISTOLOGY CSV
Merged_HE_10_class_dataset = pd.read_csv('/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/Merged_HE_10_class_dataset.csv')
Merged_KI67_7_class_dataset = pd.read_csv('/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/Merged_KI67_7_class_dataset.csv')

Merged_HE_case_ids = set(Merged_HE_10_class_dataset['case_id'])
Merged_KI67_case_ids = set(Merged_KI67_7_class_dataset['case_id'])

ids_to_append = Merged_KI67_7_class_dataset[~Merged_KI67_7_class_dataset['case_id'].isin(Merged_HE_case_ids)]

Merged_Histology_10_class_dataset = pd.concat([Merged_HE_10_class_dataset, ids_to_append], ignore_index=True)
Merged_Histology_10_class_dataset.to_csv('/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_10_class_dataset.csv', index=False)

# %%
