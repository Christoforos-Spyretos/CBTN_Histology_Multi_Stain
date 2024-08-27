# %%
import numpy as np
import h5py
import os
import torch

# %%
file_path = '/local/data1/chrsp39/CBTN_v2/extracted_mag20x_patch256_fp/patches3/C23862___7316-101___HandE_BLOCK_A1.h5'

# Open the HDF5 file in read-only mode
with h5py.File(file_path, 'r') as file:
    # Print the keys at the root level
    print("Keys:", list(file.keys()))

    # Choose a dataset key from the list above
    dataset_key = 'coords'

    # Get the shape and content of a specific dataset
    if dataset_key in file:
        dataset = file[dataset_key]
        print(dataset.shape)
        print(dataset[:])  # Print the entire content, you may want to adjust this based on your dataset size

        root_attrs = file.attrs
        print("Attributes at root level:", list(root_attrs.keys()))
    else:
        print(f"Dataset key '{dataset_key}' not found in the file.")

# %%
import torch

# Replace 'your_file.pt' with the path to your .pt file
file_path = '/local/data1/chrsp39/CBTN_v2/extracted_mag20x_patch256_fp/resnet50_trunc_pt_patch_features/pt_files/C27552___7316-314___HandE_3.pt'
loaded_data = torch.load(file_path)

if isinstance(loaded_data, torch.Tensor):
    print("Tensor dimensions:", loaded_data.size())

# %%
checkpoints_path = '/home/chrsp39/Cross_modal_data_fusion/models/CLAM/results/HE_tumour_subyting_initial_attempt_s1/s_0_checkpoint.pt'

checkpoint = torch.load(checkpoints_path)

print(checkpoint.keys())

# %%
file_path = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/merged_resnet_features/h5_files/C773424.h5'

# Open the HDF5 file in read-only mode
with h5py.File(file_path, 'r') as file:
    print("Keys:", list(file.keys()))
    print('Coords')
    coords = file['coords']
    print(coords.shape)
    print(coords[:])
    print('Features')
    features = file['features']
    print(features.shape)
    print(features[:])

# %%
pt_files = '/local/data2/chrsp39/CBTN_v2/UNI/HE/uni_vit_features/pt_files'
tensor_files = os.listdir(pt_files)
subject_id = 'C58917___'

for tensor_file in tensor_files:
    if tensor_file.startswith(subject_id):
        tensor_path = os.path.join(pt_files, tensor_file)
        tensor = torch.load(tensor_path)
        print(tensor.shape)

pt_file = torch.load("/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE/uni_vit_features/pt_files/C58917.pt")
print(pt_file.shape)

# %%
pt_file = torch.load("/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE/uni_vit_features/pt_files/C58917.pt")
print(pt_file.shape)

pt_file = torch.load("/local/data2/chrsp39/CBTN_v2/UNI/Merged_KI67/uni_vit_features/pt_files/C58917.pt")
print(pt_file.shape)

pt_file = torch.load("/local/data2/chrsp39/CBTN_v2/UNI/Merged_GFAP/uni_vit_features/pt_files/C58917.pt")
print(pt_file.shape)

pt_file = torch.load("/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67_GFAP/uni_vit_features/pt_files/C58917.pt")
print(pt_file.shape)

# %%
import pandas as pd 

Histology_bounded = pd.read_csv('/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_7_class_dataset_bounded.csv')
Histology = pd.read_csv('/local/data2/chrsp39/CBTN_v2/CLAM/Merged_Histology/Merged_Histology_7_class_dataset.csv')

# %% 
from PIL import Image
import matplotlib.pyplot as plt

image_path = '/local/data2/chrsp39/CBTN_v2/HE/masks/C36654___7316-149___HandE.jpg'
image = Image.open(image_path)

width, height = image.size

dpi =200

fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

plt.imshow(image)
plt.axis('off')  
plt.show()


# %%
import openslide
from PIL import Image
import matplotlib.pyplot as plt

HE_slide = openslide.OpenSlide('/local/data2/chrsp39/CBTN_v2/HE/WSI/C36654___7316-149___HandE.svs')
# KI67_slide = openslide.OpenSlide('/local/data2/chrsp39/CBTN_v2/KI67/WSI/C1031970___7316-4208___Ki-67.svs')
# GFAP_slide = openslide.OpenSlide('/local/data2/chrsp39/CBTN_v2/GFAP/WSI/C1031970___7316-4208___GFAP.svs')

HE_dimensions = HE_slide.dimensions
# KI67_dimensions = KI67_slide.dimensions
# GFAP_dimensions = GFAP_slide.dimensions

HE_region = HE_slide.read_region((0, 0), 0, HE_dimensions)
# KI67_region = KI67_slide.read_region((0, 0), 0, KI67_dimensions)
# GFAP_region = GFAP_slide.read_region((0, 0), 0, GFAP_dimensions)

HE_region = HE_region.convert("RGB")
# KI67_region = KI67_region.convert("RGB")
# GFAP_region = GFAP_region.convert("RGB")

fig1, ax1 = plt.subplots(figsize=(15, 15))
ax1.imshow(HE_region)
ax1.axis('off')
plt.show()

# fig2, ax2 = plt.subplots(figsize=(15, 15))
# ax2.imshow(KI67_region)
# ax2.axis('off')
# plt.show()

# fig3, ax3 = plt.subplots(figsize=(15, 15))
# ax3.imshow(GFAP_region)
# ax3.axis('off')
# plt.show()

# %%
import pandas as pd

# df_HE = [
    # {'case_id': 'C663093', 'slide_id':'C663093___7316-2572___HandE', 'label': 'ASTR_LGG'},
    # {'case_id': 'C96432', 'slide_id':'C96432___7316-461___HandE', 'label': 'ASTR_HGG'},
    # {'case_id': 'C17589', 'slide_id':'C17589___7316-5___HandE', 'label': 'MED'},
    # {'case_id': 'C38991', 'slide_id':'C38991___7316-175___HandE', 'label': 'ASTR_LGG'},
    # {'case_id': 'C139851', 'slide_id':'C139851___7316-736___HandE_E', 'label': 'EP'},
#     {'case_id': 'C154980', 'slide_id':'C154980___7316-960___HandE', 'label': 'ASTR_LGG'}
#     ]

# df_HE = pd.DataFrame(df_HE)
# df_HE.to_csv('/local/data2/chrsp39/CBTN_v2/ATTENTION_MAPS/HE_LGG_HGG_attention_maps.csv')

df_KI67 = [
    # {'case_id': 'C663093', 'slide_id':'C663093___7316-2572___KI-67', 'label': 'ASTR_LGG'},
    # {'case_id': 'C96432', 'slide_id':'C96432___7316-461___Ki-67', 'label': 'ASTR_HGG'},
    # {'case_id': 'C17589', 'slide_id':'C17589___7316-5___KI67', 'label': 'MED'},
    # {'case_id': 'C56826', 'slide_id':'C56826___7316-195___Ki-67', 'label': 'ASTR_HGG'},
    # {'case_id': 'C57810', 'slide_id':'C57810___7316-201___Ki-67', 'label': 'ASTR_LGG'},
    # {'case_id': 'C2760858', 'slide_id':'C2760858___7316-6778___Ki-67', 'label': 'ASTR_LGG'},
    # {'case_id': 'C154980', 'slide_id':'C154980___7316-960___Ki-67', 'label': 'ASTR_LGG'},
    # {'case_id': 'C524349', 'slide_id':'C524349___7316-2152___Ki-67', 'label': 'ASTR_HGG'},
    {'case_id': 'C39483', 'slide_id':'C39483___7316-183___Ki-67', 'label': 'ASTR_LGG'},
    {'case_id': 'C60270', 'slide_id':'C60270___7316-199___Ki-67', 'label': 'ASTR_LGG'},
    {'case_id': 'C99753', 'slide_id':'C99753___7316-466___Ki-67', 'label': 'ASTR_HGG'},
    {'case_id': 'C121524', 'slide_id':'C121524___7316-495___Ki-67', 'label': 'ASTR_LGG'},
    {'case_id': 'C377856', 'slide_id':'C377856___7316-1763___Ki-67_B1', 'label': 'ASTR_HGG'},
    {'case_id': 'C410328', 'slide_id':'C410328___7316-1097___Ki-67', 'label': 'ASTR_LGG'},
    {'case_id': 'C524349', 'slide_id':'C524349___7316-2152___Ki-67', 'label': 'ASTR_HGG'},
    {'case_id': 'C714384', 'slide_id':'C714384___7316-3058___Ki-67', 'label': 'ASTR_HGG'},
    {'case_id': 'C868872', 'slide_id':'C868872___7316-3308___Ki-67_B1', 'label': 'ASTR_LGG'},
    {'case_id': 'C1046853', 'slide_id':'C1046853___7316-4335___Ki-67_B1', 'label': 'ASTR_HGG'},
    ]

df_KI67 = pd.DataFrame(df_KI67)
df_KI67.to_csv('/local/data2/chrsp39/CBTN_v2/ATTENTION_MAPS/KI67_LGG_HGG_attention_maps.csv')

# df_GFAP = [
    # {'case_id': 'C663093', 'slide_id':'C663093___7316-2572___GFAP', 'label': 'ASTR_LGG'},
    # {'case_id': 'C96432', 'slide_id':'C96432___7316-461___GFAP', 'label': 'ASTR_HGG'},
    # # {'case_id': 'C17589', 'slide_id':'C17589___7316-5___GFAP', 'label': 'MED'},
    # {'case_id': 'C56826', 'slide_id':'C56826___7316-195___GFAP', 'label': 'ASTR_HGG'},
    # {'case_id': 'C57810', 'slide_id':'C57810___7316-201___GFAP', 'label': 'ASTR_LGG'},
    # {'case_id': 'C2760858', 'slide_id':'C2760858___7316-6778___GFAP', 'label': 'ASTR_LGG'},
    # {'case_id': 'C154980', 'slide_id':'C154980___7316-960___GFAP', 'label': 'ASTR_LGG'},
    # {'case_id': 'C524349', 'slide_id':'C524349___7316-2152___GFAP', 'label': 'ASTR_HGG'}
    # ]

# df_GFAP = pd.DataFrame(df_GFAP)
# df_GFAP.to_csv('/local/data2/chrsp39/CBTN_v2/ATTENTION_MAPS/GFAP_LGG_HGG_attention_maps.csv')

# %%
import torch
import matplotlib.pyplot as plt

def min_max_normalize(tensor):
    min_val = torch.min(tensor, dim=0, keepdim=True).values
    max_val = torch.max(tensor, dim=0, keepdim=True).values
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def plot_histograms(original, normalized):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

    axs[0].hist(original.numpy().flatten(), bins=50, alpha=0.7)
    axs[0].set_title('Original Features Histogram')
    axs[0].set_xlabel('Feature Value')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(normalized.numpy().flatten(), bins=50, alpha=0.7)
    axs[1].set_title('Normalized Features Histogram')
    axs[1].set_xlabel('Feature Value')
    axs[1].set_ylabel('Frequency')

    plt.show()

# %%
features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE/uni_vit_features/pt_files/C41574.pt')
normalized_features = min_max_normalize(features)
plot_histograms(features, normalized_features)

# %%
features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_KI67/uni_vit_features/pt_files/C41574.pt')
normalized_features = min_max_normalize(features)
plot_histograms(features, normalized_features)

# %%
features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_GFAP/uni_vit_features/pt_files/C41574.pt')
normalized_features = min_max_normalize(features)
plot_histograms(features, normalized_features)

# %%
features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67/uni_vit_features/pt_files/C41574.pt')
normalized_features = min_max_normalize(features)
plot_histograms(features, normalized_features)

# %%
features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_GFAP/uni_vit_features/pt_files/C41574.pt')
normalized_features = min_max_normalize(features)
plot_histograms(features, normalized_features)

# %%
features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE_KI67_GFAP/uni_vit_features/pt_files/C41574.pt')
normalized_features = min_max_normalize(features)
plot_histograms(features, normalized_features)

# %%
import torch
import matplotlib.pyplot as plt
import umap

he_features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_HE/uni_vit_features/pt_files/C99753.pt')
ki67_features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_KI67/uni_vit_features/pt_files/C99753.pt')
gfap_features = torch.load('/local/data2/chrsp39/CBTN_v2/UNI/Merged_GFAP/uni_vit_features/pt_files/C99753.pt')

normalized_he_features = min_max_normalize(he_features)
normalized_ki67_features = min_max_normalize(ki67_features)
normalized_gfap_features = min_max_normalize(gfap_features)

def plot_umap(features_list, labels_list, title):
    plt.figure(figsize=(10, 8))
    reducer = umap.UMAP()
    for features, label in zip(features_list, labels_list):
        features_np = features.numpy()
        embedding = reducer.fit_transform(features_np)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, label=label)
    plt.legend()
    plt.title(title)
    plt.show()

plot_umap(
    [normalized_he_features, normalized_ki67_features, normalized_gfap_features],
    ['Normalized HE Features', 'Normalized KI67 Features', 'Normalized GFAP Features'],
    "UMAP of Normalized HE, KI67, and GFAP Features"
)

# %%
import os
import shutil
import pandas as pd

images_path = '/local/data2/chrsp39/CBTN_v2/KI67/WSI'
df = pd.read_csv('/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Histology_5_Merged_HE_KI67_GFAP_class_bounded_small_clam_sb_uni_vit/fold_44.csv')
save_path = '/run/media/chrsp39/SD/CBTN/KI67'

images = os.listdir(images_path)

for subject_id in df['slide_id']:
    for image in images:
        if image.startswith(subject_id + '___'):
            src_path = os.path.join(images_path, image)
            dst_path = os.path.join(save_path, image)
            shutil.copy(src_path, dst_path)

# %%
images_path = '/local/data2/chrsp39/CBTN_v2/GFAP/WSI'
df = pd.read_csv('/home/chrsp39/CBTN_Histology_Multi_Modal/models/CLAM/eval_results/EVAL_Merged_GFAP_LGG_vs_HGG_bounded_clam_sb_small_uni_vit/fold_42.csv')
save_path = '/run/media/chrsp39/SD/CBTN/GFAP'

images = os.listdir(images_path)

for subject_id in df['slide_id']:
    for image in images:
        if image.startswith(subject_id + '___'):
            src_path = os.path.join(images_path, image)
            dst_path = os.path.join(save_path, image)
            shutil.copy(src_path, dst_path)

# %% IMPORTS
import pandas as pd
import os

# %% LOAD DATA
histology_images = "/run/media/chrsp39/CBNT_v2/Ki-67/WSI"
# histology_df = pd.read_csv("/run/media/chrsp39/CBNT_v2/Ki-67/CSV_FILES/CBTN_histology_summary.csv")
histology_df = pd.read_csv("/local/data2/chrsp39/CBTN_v2/CBTN_histology_summary.csv")

# %% CLEAN HISTOLOGY DATAFRAME
histology_df = histology_df.loc[histology_df['image_type'] == "KI-67"]

histology_df = histology_df[['subjectID','diagnosis','survival', 'image_type']]

tumour_types = [
    'Low-grade glioma/astrocytoma (WHO grade I/II)',
    'High-grade glioma/astrocytoma (WHO grade III/IV)', 
    'Medulloblastoma',
    'Ependymoma',
    'Brainstem glioma- Diffuse intrinsic pontine glioma',
    'Atypical Teratoid Rhabdoid Tumor (ATRT)',
    'Meningioma',
    # 'Craniopharyngioma',
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
    # 'Craniopharyngioma': 'CRAN',
    'Dysembryoplastic neuroepithelial tumor (DNET)': 'DNET',
    'Ganglioglioma': 'GANG'
})

# %%
df_csv_list = []
subjects = os.listdir(histology_images)

for subject in subjects:
    file_name = os.path.splitext(subject)[0] 
    case_id = file_name.split("___")[0]
    slide_id = "___".join(file_name.split("___")[0:])
    if case_id in histology_df['subjectID'].values:
        label = histology_df.loc[histology_df['subjectID'] == case_id, 'diagnosis'].values[0]
        survival = histology_df.loc[histology_df['subjectID'] == case_id, 'survival'].values[0]
        df_csv_list.append({'case_id': case_id, 'slide_id': slide_id, 'label': label, 'survival': survival})

dataset_csv = pd.DataFrame(df_csv_list)

dataset_csv.to_csv('/run/media/chrsp39/CBNT_v2/Ki-67/CSV_FILES/Ki-67_summary.csv', index=False)

# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df_KI67 = pd.read_csv("/run/media/chrsp39/CBNT_v2/Ki-67/CSV_FILES/Ki-67_summary.csv")

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
# %%
