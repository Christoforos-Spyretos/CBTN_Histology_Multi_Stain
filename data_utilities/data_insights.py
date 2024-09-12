# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# %% H&E
df_HE = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSV_FILES/FEATURE_EXTRACTION/HE/HE_10_class_dataset.csv")

subjects_per_tumour_HE = df_HE.groupby(['case_id'])['label'].min().reset_index()
sorted_subjects_per_tumour_HE = subjects_per_tumour_HE.iloc [:,1].value_counts().sort_values(ascending=False)
subjects_per_tumour_HE = pd.DataFrame(sorted_subjects_per_tumour_HE).reset_index()

sorted_images_per_tumour_HE = df_HE.iloc [:,2].value_counts().sort_values(ascending=False)
images_per_tumour_HE = pd.DataFrame(sorted_images_per_tumour_HE).reset_index()

merged_df_HE = pd.merge(subjects_per_tumour_HE, images_per_tumour_HE, on='label', suffixes=('_subjects', '_images'))
merged_df_HE = merged_df_HE.sort_values(by='label')
merged_df_HE.columns = ['Label', 'Number of Subjects', 'Number of Images']
merged_df_HE = merged_df_HE.sort_values(by='Number of Subjects',ascending=False)
print('H&E information')
print(f'Total number of subjects: {merged_df_HE["Number of Subjects"].sum()}')
print(f'Total number of slides: {merged_df_HE["Number of Images"].sum()}')
print(merged_df_HE)

bar_width = 0.4
x = np.arange(len(merged_df_HE['Label']))

fig = plt.figure(figsize=(12, 8))
plt.bar(x - bar_width/2, merged_df_HE['Number of Subjects'], color='grey', width=0.4, label='Number of subjects')
for i, count in enumerate(merged_df_HE['Number of Subjects']):
    plt.text(i - bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + bar_width/2, merged_df_HE['Number of Images'], color='maroon', width=0.4, label='Number of slides')
for i, count in enumerate(merged_df_HE['Number of Images']):
    plt.text(i + bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour family/type")
plt.ylabel("Count")
plt.title("Number subjects and slides with H&E stain per tumour family/type")
plt.xticks(x, merged_df_HE['Label'])
plt.legend()
plt.tight_layout()
plt.show()

# %% KI-67
df_KI67 = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSV_FILES/FEATURE_EXTRACTION/KI67/KI67_10_class_dataset.csv")

subjects_per_tumour_KI67 = df_KI67.groupby(['case_id'])['label'].min().reset_index()
sorted_subjects_per_tumour_KI67 = subjects_per_tumour_KI67.iloc[:, 1].value_counts().sort_values(ascending=False)
subjects_per_tumour_KI67 = pd.DataFrame(sorted_subjects_per_tumour_KI67).reset_index()

sorted_images_per_tumour_K67= df_KI67.iloc[:, 2].value_counts().sort_values(ascending=False)
images_per_tumour_KI67 = pd.DataFrame(sorted_images_per_tumour_K67).reset_index()

merged_df_KI67 = pd.merge(subjects_per_tumour_KI67, images_per_tumour_KI67, on='label', suffixes=('_subjects', '_images'))
merged_df_KI67 = merged_df_KI67.sort_values(by='label')
merged_df_KI67.columns = ['Label', 'Number of Subjects', 'Number of Images']
merged_df_KI67 = merged_df_KI67.sort_values(by='Number of Subjects',ascending=False)
print('KI-67 information')
print(f'Total number of subjects: {merged_df_KI67["Number of Subjects"].sum()}')
print(f'Total number of slides: {merged_df_KI67["Number of Images"].sum()}')
# print(merged_df_KI67)

markdown_table = merged_df_KI67.to_markdown(index=False)

print(markdown_table)

bar_width = 0.4
x = np.arange(len(merged_df_KI67['Label']))

fig = plt.figure(figsize=(12, 8))
plt.bar(x - bar_width/2, merged_df_KI67['Number of Subjects'], color='gray', width=0.4, label='Number of subjects')
for i, count in enumerate(merged_df_KI67['Number of Subjects']):
    plt.text(i - bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + bar_width/2, merged_df_KI67['Number of Images'], color='darkcyan', width=0.4, label='Number of slides')
for i, count in enumerate(merged_df_KI67['Number of Images']):
    plt.text(i + bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour family/type")
plt.ylabel("Count")
plt.title("Number of subjects and slides with KI-67 stain per tumour family/type")
plt.xticks(x, merged_df_KI67['Label'])
plt.legend()
plt.tight_layout()
plt.show()

# %% GFAP
df_GFAP = pd.read_csv("/local/data3/chrsp39/CBTN_v2/CSV_FILES/FEATURE_EXTRACTION/GFAP/GFAP_10_class_dataset.csv")

subjects_per_tumour_GFAP = df_GFAP.groupby(['case_id'])['label'].min().reset_index()
sorted_subjects_per_tumour_GAP = subjects_per_tumour_GFAP.iloc[:, 1].value_counts().sort_values(ascending=False)
subjects_per_tumour_GFAP = pd.DataFrame(sorted_subjects_per_tumour_GAP).reset_index()

sorted_images_per_tumour_GFAP= df_GFAP.iloc[:, 2].value_counts().sort_values(ascending=False)
images_per_tumour_GFAP = pd.DataFrame(sorted_images_per_tumour_GFAP).reset_index()

merged_df_GFAP = pd.merge(subjects_per_tumour_GFAP, images_per_tumour_GFAP, on='label', suffixes=('_subjects', '_images'))
merged_df_GFAP = merged_df_GFAP.sort_values(by='label')
merged_df_GFAP.columns = ['Label', 'Number of Subjects', 'Number of Images']
merged_df_GFAP = merged_df_GFAP.sort_values(by='Number of Subjects',ascending=False)
print('GFAP information')
print(f'Total number of subjects: {merged_df_GFAP["Number of Subjects"].sum()}')
print(f'Total number of slides: {merged_df_GFAP["Number of Images"].sum()}')
print(merged_df_GFAP)

bar_width = 0.4
x = np.arange(len(merged_df_GFAP['Label']))

fig = plt.figure(figsize=(12, 8))
plt.bar(x - bar_width/2, merged_df_GFAP['Number of Subjects'], color='gray', width=0.4, label='Number of subjects')
for i, count in enumerate(merged_df_GFAP['Number of Subjects']):
    plt.text(i - bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + bar_width/2, merged_df_GFAP['Number of Images'], color='peru', width=0.4, label='Number of slides')
for i, count in enumerate(merged_df_GFAP['Number of Images']):
    plt.text(i + bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour family/type")
plt.ylabel("Count")
plt.title("Number subjects and slides with GFAP stain per tumour family/type")
plt.xticks(x, merged_df_GFAP['Label'])
plt.legend()
plt.tight_layout()
plt.show()

# %%
HE_subjects = set(df_HE['case_id'].values)
KI67_subjects = set(df_KI67['case_id'].values)

common_subjects = HE_subjects & KI67_subjects 
df_common = pd.DataFrame(list(common_subjects), columns=['case_id'])

df_HE_unique = df_HE.drop_duplicates(subset=['case_id'])
df_common = df_common.merge(df_HE_unique[['case_id', 'label']], on='case_id', how='left')

he_counts = df_HE.groupby('case_id').size().reset_index(name='Number of H&E Images')
ki67_counts = df_KI67.groupby('case_id').size().reset_index(name='Number of KI-67 Images')

df_common = df_common.merge(he_counts, on='case_id', how='left')
df_common = df_common.merge(ki67_counts, on='case_id', how='left')

df_common[['Number of H&E Images', 'Number of KI-67 Images']] = df_common[['Number of H&E Images', 'Number of KI-67 Images']].fillna(0)

merged_histology_common = df_common.groupby('label').agg({
    'case_id': 'count',
    'Number of H&E Images': 'sum',
    'Number of KI-67 Images': 'sum',
}).reset_index()

merged_histology_common.columns = ['Label', 'Number of Subjects', 'Number of H&E Images', 'Number of KI-67 Images']

merged_histology_common = merged_histology_common.sort_values(by='Number of Subjects', ascending=False)

print(f'Total number of subjects: {merged_histology_common["Number of Subjects"].sum()}')
print(f'Total number of H&E slides: {merged_histology_common["Number of H&E Images"].sum()}')
print(f'Total number of KI-67 slides: {merged_histology_common["Number of KI-67 Images"].sum()}')
print(merged_histology_common)

bar_width = 0.2
x = np.arange(len(merged_histology_common['Label']))

fig = plt.figure(figsize=(15, 8))

plt.bar(x - 1.5 * bar_width, merged_histology_common['Number of Subjects'], color='gray', width=bar_width, label='Number of Subjects')
for i, count in enumerate(merged_histology_common['Number of Subjects']):
    plt.text(i - 1.5 * bar_width, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x - 0.5 * bar_width, merged_histology_common['Number of H&E Images'], color='maroon', width=bar_width, label='Number of H&E slides')
for i, count in enumerate(merged_histology_common['Number of H&E Images']):
    plt.text(i - 0.5 * bar_width, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + 0.5 * bar_width, merged_histology_common['Number of KI-67 Images'], color='darkcyan', width=bar_width, label='Number of KI-67 slides')
for i, count in enumerate(merged_histology_common['Number of KI-67 Images']):
    plt.text(i + 0.5 * bar_width, count, str(count), ha='center', va='bottom', fontsize=10)

plt.xlabel("Tumour family/type")
plt.ylabel("Count")
plt.title("Number of subjects with H&E and KI-67 stain slides per tumour family/type")
plt.xticks(x, merged_histology_common['Label'])
plt.legend()
plt.tight_layout()
plt.show()

# %%
HE_subjects = set(df_HE['case_id'].values)
KI67_subjects = set(df_KI67['case_id'].values)
GFAP_subjects = set(df_GFAP['case_id'].values)

common_subjects = HE_subjects & KI67_subjects & GFAP_subjects
df_common = pd.DataFrame(list(common_subjects), columns=['case_id'])

df_HE_unique = df_HE.drop_duplicates(subset=['case_id'])
df_common = df_common.merge(df_HE_unique[['case_id', 'label']], on='case_id', how='left')

he_counts = df_HE.groupby('case_id').size().reset_index(name='Number of H&E Images')
ki67_counts = df_KI67.groupby('case_id').size().reset_index(name='Number of KI-67 Images')
gfap_counts = df_GFAP.groupby('case_id').size().reset_index(name='Number of GFAP Images')

df_common = df_common.merge(he_counts, on='case_id', how='left')
df_common = df_common.merge(ki67_counts, on='case_id', how='left')
df_common = df_common.merge(gfap_counts, on='case_id', how='left')

df_common[['Number of H&E Images', 'Number of KI-67 Images', 'Number of GFAP Images']] = df_common[['Number of H&E Images', 'Number of KI-67 Images', 'Number of GFAP Images']].fillna(0)

merged_histology_common = df_common.groupby('label').agg({
    'case_id': 'count',
    'Number of H&E Images': 'sum',
    'Number of KI-67 Images': 'sum',
    'Number of GFAP Images': 'sum'
}).reset_index()

merged_histology_common.columns = ['Label', 'Number of Subjects', 'Number of H&E Images', 'Number of KI-67 Images', 'Number of GFAP Images']

merged_histology_common = merged_histology_common.sort_values(by='Number of Subjects', ascending=False)

print(f'Total number of subjects: {merged_histology_common["Number of Subjects"].sum()}')
print(f'Total number of H&E slides: {merged_histology_common["Number of H&E Images"].sum()}')
print(f'Total number of KI-67 slides: {merged_histology_common["Number of KI-67 Images"].sum()}')
print(f'Total number of GFAP slides: {merged_histology_common["Number of GFAP Images"].sum()}')
print(merged_histology_common)

bar_width = 0.2
x = np.arange(len(merged_histology_common['Label']))

fig = plt.figure(figsize=(15, 8))

plt.bar(x - 1.5 * bar_width, merged_histology_common['Number of Subjects'], color='gray', width=bar_width, label='Number of Subjects')
for i, count in enumerate(merged_histology_common['Number of Subjects']):
    plt.text(i - 1.5 * bar_width, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x - 0.5 * bar_width, merged_histology_common['Number of H&E Images'], color='maroon', width=bar_width, label='Number of H&E slides')
for i, count in enumerate(merged_histology_common['Number of H&E Images']):
    plt.text(i - 0.5 * bar_width, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + 0.5 * bar_width, merged_histology_common['Number of KI-67 Images'], color='darkcyan', width=bar_width, label='Number of KI-67 slides')
for i, count in enumerate(merged_histology_common['Number of KI-67 Images']):
    plt.text(i + 0.5 * bar_width, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + 1.5 * bar_width, merged_histology_common['Number of GFAP Images'], color='peru', width=bar_width, label='Number of GFAP slides')
for i, count in enumerate(merged_histology_common['Number of GFAP Images']):
    plt.text(i + 1.5 * bar_width, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour family/type")
plt.ylabel("Count")
plt.title("Number of subjects with H&E, KI-67, and GFAP stain slides per tumour family/type")
plt.xticks(x, merged_histology_common['Label'])
plt.legend()
plt.tight_layout()
plt.show()

# %%
