# %% IMPORTS
import random
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# %% LOAD CSV FILE AND DATA PATH 
df = pd.read_csv(r'/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/SUMMARY_FILES/HISTOLOGY/CBTN_histology_summary.csv')
histology_path = '/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/SUBJECTS'

# %% PREPARING SUBJECTS 
df1 = df.loc[df['subjectID'] != 'Not_available']
df1 = df.drop_duplicates(subset=["subjectID"], keep=False) 

# stain types to be selected
stain_types = [
    'H&E'
    ]

df2 = df1[df1['image_type'].isin(stain_types)]

# tumors to be selected
tumor_types = [
    'Low-grade glioma/astrocytoma (WHO grade I/II)',
    'High-grade glioma/astrocytoma (WHO grade III/IV)', 
    'Ependymoma',
    'Brainstem glioma- Diffuse intrinsic pontine glioma',
    'Medulloblastoma',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)'
    ]
    
df2 = df2[df2['diagnosis'].isin(tumor_types)]

# rename tumour types
df2['diagnosis'] = df2['diagnosis'].replace({
    'Low-grade glioma/astrocytoma (WHO grade I/II)': 'ASTR_LGG',
    'High-grade glioma/astrocytoma (WHO grade III/IV)' : 'ASTR_HGG',
    'Ependymoma' : 'EP',
    'Medulloblastoma': 'MED',
    'Brainstem glioma- Diffuse intrinsic pontine glioma': 'DIPG',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)': 'ATRT'
})

df2 = df2[['subjectID', 'diagnosis', 'image_type']]

# %% SPLIT PATIENTS INTO TRAINING, VALIDATION AND TEST SETS (80%, 10%, 10%)
subjectIDs = list(df2['subjectID'].unique())
random.shuffle(subjectIDs)

# %%
train, test = train_test_split(subjectIDs, test_size=0.2)
val, test = train_test_split(test, test_size=0.5)

# storing the split sets
split_data = {
    'train': train,
    'validation': val,
    'test': test
}

# save split info into csv
split_df = pd.DataFrame(columns=['subjectID', 'split_set', 'diagnosis'])

for split_set, subject_list in split_data.items():
    temp_df = pd.DataFrame({'subjectID': subject_list, 'split_set': split_set})
    temp_df = temp_df.merge(df2[['subjectID', 'diagnosis']], on='subjectID', how='left')
    split_df = pd.concat([split_df, temp_df])

split_df.to_csv('split_info.csv', index=False)

# save the split sets to a json file
with open('split_sets.json', 'w') as json_file:
    json.dump(split_data, json_file)

# load the json file
with open('split_sets.json', 'r') as json_file:
    split_data = json.load(json_file)

train = [subjectID for subjectID in subjectIDs if subjectID in split_data['train']]
val = [subjectID for subjectID in subjectIDs if subjectID in split_data['validation']]
test = [subjectID for subjectID in subjectIDs if subjectID in split_data['test']]

# %%
