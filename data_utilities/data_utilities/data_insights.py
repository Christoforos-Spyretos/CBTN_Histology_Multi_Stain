# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# %% H&E
df_HE = pd.read_csv("/local/data2/chrsp39/CBTN_v2/CLAM/HE/HE_6_class_dataset.csv")

images_HE = len(df_HE)

unique_subjects_HE= df_HE['case_id'].nunique()

subjects_per_tumour_HE = df_HE.groupby(['case_id'])['label'].min().reset_index()
sorted_subjects_per_tumour_HE = subjects_per_tumour_HE.iloc [:,1].value_counts().sort_values(ascending=False)
subjects_per_tumour_HE = pd.DataFrame(sorted_subjects_per_tumour_HE).reset_index()
subjects_per_tumour_HE.rename(columns={'index':'label','label':'count'}, inplace=True)


sorted_images_per_tumour_HE = df_HE.iloc [:,2].value_counts().sort_values(ascending=False)
images_per_tumour_HE = pd.DataFrame(sorted_images_per_tumour_HE).reset_index()
images_per_tumour_HE.rename(columns={'index':'label','label':'count'}, inplace=True)

print("H&E data insights")
print(f"Total number of unique subjects: {unique_subjects_HE}")
for index, row in subjects_per_tumour_HE.iterrows():
    print(f'Total number of subjects of {row["label"]}: {row["count"]}')

print('---------------------------------------------------------------')

print(f"Total number of images: {images_HE}")
for index, row in images_per_tumour_HE.iterrows():
    print(f'Total number of images of {row["label"]}: {row["count"]}')

# fig = plt.figure(figsize = (12, 8))
# plt.bar(subjects_per_tumour_HE['label'], subjects_per_tumour_HE['count'], color ='maroon', width = 0.4)
# for i, count in enumerate(subjects_per_tumour_HE['count']):
#     plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
# plt.xlabel("Tumour type")
# plt.ylabel("Number of subjects")
# plt.title("Number of subjects per tumour type (H&E)")
# plt.show()

# fig = plt.figure(figsize = (12, 8))
# plt.bar(images_per_tumour_HE['label'], images_per_tumour_HE['count'], color ='darkgreen', width = 0.4)
# for i, count in enumerate(images_per_tumour_HE['count']):
#     plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
# plt.xlabel("Tumour type")
# plt.ylabel("Number of slides")
# plt.title("Number of slides per tumour type (H&E)")
# plt.show()

bar_width = 0.4
x = np.arange(len(subjects_per_tumour_HE['label']))

fig = plt.figure(figsize=(12, 8))
plt.bar(x - bar_width/2, subjects_per_tumour_HE['count'], color='maroon', width=0.4, label='Number of subjects')
for i, count in enumerate(subjects_per_tumour_HE['count']):
    plt.text(i - bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + bar_width/2, images_per_tumour_HE['count'], color='darkgreen', width=0.4, label='Number of slides')
for i, count in enumerate(images_per_tumour_HE['count']):
    plt.text(i + bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour type")
plt.ylabel("Count")
plt.title("Number of subjects with slides and slides per tumour type (H&E)")
plt.xticks(x, subjects_per_tumour_HE['label'])
plt.legend()
plt.tight_layout()
plt.show()

# %% KI-67
df_KI67 = pd.read_csv("/local/data2/chrsp39/CBTN_v2/CLAM/KI67/KI67_6_class_dataset.csv")

images_KI67 = len(df_KI67)

unique_subjects_KI67 = df_KI67['case_id'].nunique()

subjects_per_tumour_KI67 = df_KI67.groupby(['case_id'])['label'].min().reset_index()
sorted_subjects_per_tumour_KI67 = subjects_per_tumour_KI67.iloc [:,1].value_counts().sort_values(ascending=False)
subjects_per_tumour_KI67 = pd.DataFrame(sorted_subjects_per_tumour_KI67).reset_index()
subjects_per_tumour_KI67.rename(columns={'index':'label','label':'count'}, inplace=True)


sorted_images_per_tumour_K67= df_KI67.iloc [:,2].value_counts().sort_values(ascending=False)
images_per_tumour_KI67 = pd.DataFrame(sorted_images_per_tumour_K67).reset_index()
images_per_tumour_KI67.rename(columns={'index':'label','label':'count'}, inplace=True)

print("KI-67 data insights")
print(f"Total number of unique subjects: {unique_subjects_KI67}")
for index, row in subjects_per_tumour_KI67.iterrows():
    print(f'Total number of subjects of {row["label"]}: {row["count"]}')

print('---------------------------------------------------------------')

print(f"Total number of images: {images_KI67}")
for index, row in images_per_tumour_KI67.iterrows():
    print(f'Total number of images of {row["label"]}: {row["count"]}')

# fig = plt.figure(figsize = (12, 8))
# plt.bar(subjects_per_tumour_KI67['label'], subjects_per_tumour_KI67['count'], color ='maroon', width = 0.4)
# for i, count in enumerate(subjects_per_tumour_KI67['count']):
#     plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
# plt.xlabel("Tumour type")
# plt.ylabel("Number of subjects")
# plt.title("Number of subjects per tumour type (KI-67)")
# plt.show()

# fig = plt.figure(figsize = (12, 8))
# plt.bar(images_per_tumour_KI67['label'], images_per_tumour_KI67['count'], color ='navy', width = 0.4)
# for i, count in enumerate(images_per_tumour_KI67['count']):
#     plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
# plt.xlabel("Tumour type")
# plt.ylabel("Number of slides")
# plt.title("Number of slides per tumour type (KI-67)")
# plt.show()

bar_width = 0.4
x = np.arange(len(subjects_per_tumour_KI67['label']))

fig = plt.figure(figsize=(12, 8))
plt.bar(x - bar_width/2, subjects_per_tumour_KI67['count'], color='maroon', width=0.4, label='Number of subjects')
for i, count in enumerate(subjects_per_tumour_KI67['count']):
    plt.text(i - bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + bar_width/2, images_per_tumour_KI67['count'], color='navy', width=0.4, label='Number of slides')
for i, count in enumerate(images_per_tumour_KI67['count']):
    plt.text(i + bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour type")
plt.ylabel("Count")
plt.title("Number of subjects and slides per tumour type (KI-67)")
plt.xticks(x, subjects_per_tumour_KI67['label'])
plt.legend()
plt.tight_layout()
plt.show()

# %%
common_subjects = []

for case_id in df_KI67['case_id']:
    if case_id in df_HE['case_id'].values:
        label = df_HE.loc[df_HE['case_id'] == case_id, 'label'].values[0]
        common_subjects.append({'case_id': case_id, 'label': label})

df_common= pd.DataFrame(common_subjects)

images_common = len(df_common)

unique_subjects_common = df_common['case_id'].nunique()

subjects_per_tumour_common = df_common.groupby(['case_id'])['label'].min().reset_index()
sorted_subjects_per_tumour_common = subjects_per_tumour_common.iloc [:,1].value_counts().sort_values(ascending=False)
subjects_per_tumour_common = pd.DataFrame(sorted_subjects_per_tumour_common).reset_index()
subjects_per_tumour_common.rename(columns={'index':'label','label':'count'}, inplace=True)

sorted_images_per_tumour_common = df_common.iloc [:,1].value_counts().sort_values(ascending=False)
images_per_tumour_common = pd.DataFrame(sorted_images_per_tumour_common).reset_index()
images_per_tumour_common.rename(columns={'index':'label','label':'count'}, inplace=True)

print("H&E data insights")
print(f"Total number of unique subjects: {unique_subjects_common}")
for index, row in subjects_per_tumour_common.iterrows():
    print(f'Total number of subjects of {row["label"]}: {row["count"]}')

print('---------------------------------------------------------------')

print(f"Total number of images: {images_common}")
for index, row in images_per_tumour_common.iterrows():
    print(f'Total number of images of {row["label"]}: {row["count"]}')

# fig = plt.figure(figsize = (12, 8))
# plt.bar(subjects_per_tumour_common['label'], subjects_per_tumour_common['count'], color ='maroon', width = 0.4)
# for i, count in enumerate(subjects_per_tumour_common['count']):
#     plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
# plt.xlabel("Tumour type")
# plt.ylabel("Number of subjects")
# plt.title("Number of subjects per tumour type (H&E and KI67)")
# plt.show()

# fig = plt.figure(figsize = (12, 8))
# plt.bar(images_per_tumour_common['label'], images_per_tumour_common['count'], color ='darkcyan', width = 0.4)
# for i, count in enumerate(images_per_tumour_common['count']):
#     plt.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
# plt.xlabel("Tumour type")
# plt.ylabel("Number of slides")
# plt.title("Number of slides per tumour type (H&E and KI67)")
# plt.show()

bar_width = 0.4
x = np.arange(len(subjects_per_tumour_common['label']))

fig = plt.figure(figsize=(12, 8))
plt.bar(x - bar_width/2, subjects_per_tumour_common['count'], color='maroon', width=0.4, label='Number of subjects')
for i, count in enumerate(subjects_per_tumour_common['count']):
    plt.text(i - bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.bar(x + bar_width/2, images_per_tumour_common['count'], color='darkcyan', width=0.4, label='Number of slides')
for i, count in enumerate(images_per_tumour_common['count']):
    plt.text(i + bar_width/2, count, str(count), ha='center', va='bottom', fontsize=10)
plt.xlabel("Tumour type")
plt.ylabel("Count")
plt.title("Number of subjects with slides and slides per tumour type (H&E and KI67)")
plt.xticks(x, subjects_per_tumour_common['label'])
plt.legend()
plt.tight_layout()
plt.show()

# %%
