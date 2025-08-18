# %% IMPORTS
import pandas as pd
import os

# %% LOAD DATA
# KI67_tif = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI_tif'
# KI67_svs = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI_svs'  

HE_svs = '/run/media/chrsp39/Expansion/CBTN_v2/HE/WSI'

histology_df = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Histological Diagnoses', engine='openpyxl')

# %% CLEAN HISTOLOGY DATAFRAME
histology_df = histology_df[['External Id', 
                             'External Sample Id',
                             'Histological Diagnosis (Mondo)']]

histology_df = histology_df.rename(columns={
    'External Id': 'subjectID',
    'External Sample Id': 'sessionID',
    'Histological Diagnosis (Mondo)': 'diagnosis'
})

# %%
tumour_types = [
    'grade III glioma (MONDO:0021640)',
    'low grade glioma (MONDO:0021637)', 
    'medulloblastoma (MONDO:0007959)',
    'ependymoma (MONDO:0016698)',
    'diffuse intrinsic pontine glioma (MONDO:0006033)',
    'atypical teratoid rhabdoid tumor (MONDO:0020560)',
    'pediatric meningioma (MONDO:0003057)',
    'craniopharyngioma (MONDO:0018907)',
    'dysembryoplastic neuroepithelial tumor (MONDO:0005505)',
    'ganglioglioma (MONDO:0016733)'
    ]
 
histology_df = histology_df[histology_df['diagnosis'].isin(tumour_types)]

histology_df['diagnosis'] = histology_df['diagnosis'].replace({
    'low grade glioma (MONDO:0021637)': 'LGG',
    'grade III glioma (MONDO:0021640)': 'HGG', 
    'medulloblastoma (MONDO:0007959)': 'MB',
    'ependymoma (MONDO:0016698)': 'EP',
    'diffuse intrinsic pontine glioma (MONDO:0006033)': 'DIPG',
    'atypical teratoid rhabdoid tumor (MONDO:0020560)': 'ATRT',
    'pediatric meningioma (MONDO:0003057)': 'MEN',
    'craniopharyngioma (MONDO:0018907)': 'CRAN',
    'dysembryoplastic neuroepithelial tumor (MONDO:0005505)': 'DNET',
    'ganglioglioma (MONDO:0016733)': 'GG'

})

# %% CREATE CSV FOR 10 CLASSES
df_csv_list = []

# subjects_svs = os.listdir(KI67_svs)
# subjects_tif = os.listdir(KI67_tif)
# subjects = subjects_svs + subjects_tif

subjects = os.listdir(HE_svs)

for subject in subjects:
    file_name = os.path.splitext(subject)[0] 
    case_id = file_name.split("___")[0]
    slide_id = "___".join(file_name.split("___")[0:])
    if case_id in histology_df['subjectID'].values:
        label = histology_df.loc[histology_df['subjectID'] == case_id, 'diagnosis'].values[0]
        df_csv_list.append({'case_id': case_id, 'slide_id': slide_id, 'label': label})
    if case_id not in histology_df['subjectID'].values:
        df_csv_list.append({'case_id': case_id, 'slide_id': slide_id, 'label': 'Not available'})

dataset_csv = pd.DataFrame(df_csv_list)

dataset_csv.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/HE_10_class_dataset.csv', index=False)

# create a CSV for svs 
# df_csv_list_svs = []
# for slide_id in subjects_svs:
#     file_name = os.path.splitext(slide_id)[0] 
#     if file_name in dataset_csv['slide_id'].values:
#         label = dataset_csv.loc[dataset_csv['slide_id'] == file_name, 'label'].values[0]
#         df_csv_list_svs.append({'case_id': case_id, 'slide_id': slide_id, 'label': label})
#     if file_name not in dataset_csv['slide_id'].values:
#         df_csv_list_svs.append({'case_id': case_id, 'slide_id': slide_id, 'label': 'Not available'})

# dataset_csv_svs = pd.DataFrame(df_csv_list_svs)
# dataset_csv_svs.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset_svs.csv', index=False)

# # create a CSV for tif
# df_csv_list_tif = []
# for slide_id in subjects_tif:
#     file_name = os.path.splitext(slide_id)[0] 
#     if file_name in dataset_csv['slide_id'].values:
#         label = dataset_csv.loc[dataset_csv['slide_id'] == file_name, 'label'].values[0]
#         df_csv_list_tif.append({'case_id': case_id, 'slide_id': slide_id, 'label': label})
#     if file_name not in dataset_csv['slide_id'].values:
#         df_csv_list_tif.append({'case_id': case_id, 'slide_id': slide_id, 'label': 'Not available'})
# dataset_csv_tif = pd.DataFrame(df_csv_list_tif)
# dataset_csv_tif.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset_tif.csv', index=False)

# %% CREATE CSV FOR SPECIFIC CLASSES
class_10_csv = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset.csv")
class_2_csv = class_10_csv.drop(class_10_csv[
    (class_10_csv['label'] == 'DNET') | 
    (class_10_csv['label'] == 'DIPG') | 
    (class_10_csv['label'] == 'CRAN') |
    (class_10_csv['label'] == 'ATRT') |
    (class_10_csv['label'] == 'MEN')  
    # (class_10_csv['label'] == 'EP')   |
    # (class_10_csv['label'] == 'GG') |
    # (class_10_csv['label'] == 'MB')
    ].index)
class_2_csv.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_5_class_dataset.csv', index=False)

# %% CREATE BOUNDED CSV 
df_HE = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/HE_5_class_dataset.csv")
df_KI67 = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/KI67_5_class_dataset.csv")

HE_subjects = set(df_HE['case_id'].values)
KI67_subjects = set(df_KI67['case_id'].values)

common_subjects = HE_subjects & KI67_subjects

df_HE_common = df_HE[df_HE['case_id'].isin(common_subjects)]
df_HE_common.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv', index=False)

# %% CREATE MERGED CSV
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_10_class_dataset.csv')

df['slide_id'] = df['slide_id'].apply(lambda x: x.split('___')[0])

merged_df = df.drop_duplicates(subset=['case_id'], keep='last')
merged_df.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_10_class_dataset.csv', index=False)

# %% MERGED HISTOLOGY CSV
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_5_class_dataset_bounded.csv')
df.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_GFAP_5_class_dataset_bounded.csv', index=False)

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
