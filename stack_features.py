# %% IMPORTS
import h5py
import torch
import os
import numpy as np
import shutil

# %% STACK SINGLE MODALITY
pt_files = '/local/data2/chrsp39/CBTN_v2/CLAM/KI67/vit_features/pt_files'
pt_output_dir = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/merged_vit_features/pt_files'

tensor_files = os.listdir(pt_files)

subjects_id = []

for tensor_file in tensor_files:
    subject_id = tensor_file.split('___')[0]
    subjects_id.append(subject_id)

subjects_id = set(subjects_id)

for subject_id in subjects_id:
    stacked_tensors = []
    tensors_to_stack = []

    for tensor_file in tensor_files:

        if tensor_file.startswith(subject_id):
            tensor_path = os.path.join(pt_files, tensor_file)
            tensor = torch.load(tensor_path)
            tensors_to_stack.append(tensor)

    stacked_tensor = torch.cat(tensors_to_stack, dim=0)
    output_file_path = os.path.join(pt_output_dir, f"{subject_id}.pt")
    torch.save(stacked_tensor, output_file_path)

    stacked_tensors = []
    tensors_to_stack = []

# %%%
# h5_filenames = os.listdir(h5_files)

# for h5_file in h5_filenames:
#     subject_id = h5_file.split('___')[0]

#     coords_list = []
#     features_list = []

#     if h5_file.startswith(subject_id):
#         h5_file_path = os.path.join(h5_files, h5_file)

#         with h5py.File(h5_file_path, 'r') as f:
#             coords = f['coords'][:]
#             features = f['features'][:]

#             coords_list.append(coords)
#             features_list.append(features)

#         coords_stacked = np.vstack(coords_list)
#         features_stacked = np.vstack(features_list)

#         output_file_path = os.path.join(h5_output_dir, f"{subject_id}.h5")

#         with h5py.File(output_file_path, 'w') as f:
#             f.create_dataset('coords', data=coords_stacked)
#             f.create_dataset('features', data=features_stacked)

# %% STACK HE & KI67 MATCHED SUBJECTS
HE_pt_files = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/merged_resnet_features/pt_files'
KI67_pt_files = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/merged_resnet_features/pt_files'
pt_output_dir = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE_KI67/merged_resnet_features/pt_files'

HE_tensor_files = os.listdir(HE_pt_files)
HE_subjects_id = []

for HE_tensor_file in HE_tensor_files:
    HE_subject_id = HE_tensor_file.split('.')[0]
    HE_subjects_id.append(HE_subject_id)

HE_subjects_id = set(HE_subjects_id)

KI67_tensor_files = os.listdir(KI67_pt_files)
KI67_subjects_id = []

for KI67_tensor_file in KI67_tensor_files:
    KI67_subject_id = KI67_tensor_file.split('.')[0]
    KI67_subjects_id.append(KI67_subject_id)

KI67_subjects_id = set(KI67_subjects_id)

for KI67_subject_id in KI67_subjects_id:
    if KI67_subject_id in HE_subjects_id:
        stacked_tensors = []
        tensors_to_stack = []

        for KI67_tensor_file in KI67_tensor_files:
            if KI67_tensor_file.startswith(KI67_subject_id):
                KI_67_tensor_path = os.path.join(KI67_pt_files, KI67_tensor_file)
                tensor = torch.load(KI_67_tensor_path)
                tensors_to_stack.append(tensor)

        for HE_tensor_file in HE_tensor_files:
            if HE_tensor_file.startswith(KI67_subject_id):
                HE_tensor_path = os.path.join(HE_pt_files, HE_tensor_file)
                tensor = torch.load(HE_tensor_path)
                tensors_to_stack.append(tensor)

        stacked_tensor = torch.cat(tensors_to_stack, dim=0)
        output_file_path = os.path.join(pt_output_dir, f"{KI67_subject_id}.pt")
        torch.save(stacked_tensor, output_file_path)

        stacked_tensors = []
        tensors_to_stack = []
        
# %% STACK ALL HE & KI67 SUBJECTS
HE_pt_files = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/merged_vit_features/pt_files'
KI67_pt_files = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/merged_vit_features/pt_files'
HE_KI67_pt_files = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE_KI67/merged_vit_features/pt_files'
pt_output_dir = '/local/data2/chrsp39/CBTN_v2/CLAM/Merged_All_HE_KI67/merged_vit_features/pt_files'

HE_KI67_tensor_files = os.listdir(HE_KI67_pt_files)

for HE_KI67_tensor_file in HE_KI67_tensor_files: 
    HE_KI67_tensor_path = os.path.join(HE_KI67_pt_files, HE_KI67_tensor_file)
    shutil.copyfile(HE_KI67_tensor_path, os.path.join(pt_output_dir, HE_KI67_tensor_file))

HE_tensor_files = os.listdir(HE_pt_files)

for HE_tensor_file in HE_tensor_files: 
    if HE_tensor_files not in HE_KI67_tensor_files:
        HE_tensor_path = os.path.join(HE_pt_files, HE_tensor_file)
        shutil.copyfile(HE_tensor_path, os.path.join(pt_output_dir, HE_tensor_file))

KI67_tensor_files = os.listdir(KI67_pt_files)

for KI67_tensor_file in KI67_tensor_files:
    if KI67_tensor_file not in HE_KI67_tensor_files:
        KI_67_tensor_path = os.path.join(KI67_pt_files, KI67_tensor_file)
        shutil.copyfile(KI_67_tensor_path, os.path.join(pt_output_dir, KI67_tensor_file))

# %%
