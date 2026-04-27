# %% IMPORTS
import pandas as pd
import os

# %% LOAD DATA
HE_svs = '/run/media/chrsp39/Expansion/CBTN_v2/HE/WSI'
KI67_tif = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI_tif'
KI67_svs = '/local/data3/chrsp39/CBTN_v2/new_KI67/WSI_svs'  

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

# %% TUMOUR TYPES TO INCLUDE
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
# create a CSV for HE (svs only)
df_csv_list = []

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

# create a CSV for KI67 (svs and tif)
subjects_svs = os.listdir(KI67_svs)
subjects_tif = os.listdir(KI67_tif)
subjects = subjects_svs + subjects_tif

# create a CSV for KI67 (svs only)
df_csv_list_svs = []
for slide_id in subjects_svs:
    file_name = os.path.splitext(slide_id)[0] 
    if file_name in dataset_csv['slide_id'].values:
        label = dataset_csv.loc[dataset_csv['slide_id'] == file_name, 'label'].values[0]
        df_csv_list_svs.append({'case_id': case_id, 'slide_id': slide_id, 'label': label})
    if file_name not in dataset_csv['slide_id'].values:
        df_csv_list_svs.append({'case_id': case_id, 'slide_id': slide_id, 'label': 'Not available'})

dataset_csv_svs = pd.DataFrame(df_csv_list_svs)
dataset_csv_svs.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset_svs.csv', index=False)

# create a CSV for KI67 (tif only)
df_csv_list_tif = []
for slide_id in subjects_tif:
    file_name = os.path.splitext(slide_id)[0] 
    if file_name in dataset_csv['slide_id'].values:
        label = dataset_csv.loc[dataset_csv['slide_id'] == file_name, 'label'].values[0]
        df_csv_list_tif.append({'case_id': case_id, 'slide_id': slide_id, 'label': label})
    if file_name not in dataset_csv['slide_id'].values:
        df_csv_list_tif.append({'case_id': case_id, 'slide_id': slide_id, 'label': 'Not available'})
dataset_csv_tif = pd.DataFrame(df_csv_list_tif)
dataset_csv_tif.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset_tif.csv', index=False)

# %% CREATE CSV FOR 5 CLASSES
# for HE
HE_10_class_csv = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/HE_10_class_dataset.csv")
HE_5_class_csv = HE_10_class_csv.drop(HE_10_class_csv[
    # (HE_10_class_csv['label'] == 'DNET') | 
    # (HE_10_class_csv['label'] == 'DIPG') | 
    # (HE_10_class_csv['label'] == 'CRAN') |
    # (HE_10_class_csv['label'] == 'ATRT') |
    # (HE_10_class_csv['label'] == 'MEN') |
    (HE_10_class_csv['label'] == 'EP')   |
    (HE_10_class_csv['label'] == 'GG') |
    (HE_10_class_csv['label'] == 'MB') |
    (HE_10_class_csv['label'] == 'HGG') |
    (HE_10_class_csv['label'] == 'LGG')
    ].index)
HE_5_class_csv.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/HE_5_class_dataset.csv', index=False)

# for KI67
KI67_10_class_csv = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset.csv")
KI67_5_class_csv = KI67_10_class_csv.drop(KI67_10_class_csv[
    # (KI67_10_class_csv['label'] == 'DNET') | 
    # (KI67_10_class_csv['label'] == 'DIPG') | 
    # (KI67_10_class_csv['label'] == 'CRAN') |
    # (KI67_10_class_csv['label'] == 'ATRT') |
    # (KI67_10_class_csv['label'] == 'MEN') |
    (KI67_10_class_csv['label'] == 'EP')   |
    (KI67_10_class_csv['label'] == 'GG') |
    (KI67_10_class_csv['label'] == 'MB') |
    (KI67_10_class_csv['label'] == 'HGG') |
    (KI67_10_class_csv['label'] == 'LGG')
    ].index)
KI67_5_class_csv.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_5_class_dataset.csv', index=False)

# create a merged HE and KI67 CSV for 5 classes
df_HE_5_classes = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/HE_5_class_dataset.csv")
df_KI67_5_classes = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/KI67_5_class_dataset.csv")

HE_5_classes_subjects = set(df_HE_5_classes['case_id'].values)
KI67_5_classes_subjects = set(df_KI67_5_classes['case_id'].values)

common_subjects_5_classes = HE_5_classes_subjects & KI67_5_classes_subjects

df_5_classes_common = df_HE_5_classes[df_HE_5_classes['case_id'].isin(common_subjects_5_classes)]
df_5_classes_common.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv', index=False)

# %% CREATE CSV FOR LGG vs HGG CLASSES
# for HE
HE_10_class_csv = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/HE_10_class_dataset.csv")
HE_2_class_csv = HE_10_class_csv.drop(HE_10_class_csv[
    # (HE_10_class_csv['label'] == 'DNET') | 
    # (HE_10_class_csv['label'] == 'DIPG') | 
    # (HE_10_class_csv['label'] == 'CRAN') |
    # (HE_10_class_csv['label'] == 'ATRT') |
    # (HE_10_class_csv['label'] == 'MEN') |
    # (HE_10_class_csv['label'] == 'EP')   |
    # (HE_10_class_csv['label'] == 'GG') |
    # (HE_10_class_csv['label'] == 'MB') |
    (HE_10_class_csv['label'] == 'HGG') |
    (HE_10_class_csv['label'] == 'LGG')
    ].index)
HE_2_class_csv.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/HE_LGG_vs_HGG_dataset.csv', index=False)

# for KI67
KI67_10_class_csv = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset.csv")
KI67_2_class_csv = KI67_10_class_csv.drop(KI67_10_class_csv[
    # (KI67_10_class_csv['label'] == 'DNET') | 
    # (KI67_10_class_csv['label'] == 'DIPG') | 
    # (KI67_10_class_csv['label'] == 'CRAN') |
    # (KI67_10_class_csv['label'] == 'ATRT') |
    # (KI67_10_class_csv['label'] == 'MEN') |
    # (KI67_10_class_csv['label'] == 'EP')   |
    # (KI67_10_class_csv['label'] == 'GG') |
    # (KI67_10_class_csv['label'] == 'MB') |
    (KI67_10_class_csv['label'] == 'HGG') |
    (KI67_10_class_csv['label'] == 'LGG')
    ].index)
KI67_2_class_csv.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_LGG_vs_HGG_dataset.csv', index=False)

# create a merged HE and KI67 CSV for LGG vs HGG classes
df_HE_LGG_vs_HGG = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/HE_LGG_vs_HGG_dataset.csv")
df_KI67_LGG_vs_HGG = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSVs/KI67_LGG_vs_HGG_dataset.csv")

HE_LGG_vs_HGG_subjects = set(df_HE_LGG_vs_HGG['case_id'].values)
KI67_LGG_vs_HGG_subjects = set(df_KI67_LGG_vs_HGG['case_id'].values)

common_subjects_LGG_vs_HGG = HE_LGG_vs_HGG_subjects & KI67_LGG_vs_HGG_subjects

df_LGG_vs_HGG_common = df_HE_LGG_vs_HGG[df_HE_LGG_vs_HGG['case_id'].isin(common_subjects_LGG_vs_HGG)]
df_LGG_vs_HGG_common.to_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_LGG_vs_HGG_dataset.csv', index=False)