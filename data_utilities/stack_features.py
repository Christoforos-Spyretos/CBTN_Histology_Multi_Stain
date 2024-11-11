# %% IMPORTS
import h5py
import torch
import os
import numpy as np
import shutil

# %% STACK SINGLE MODALITY
pt_files = '/local/data2/chrsp39/CBTN_v2/virchow2_embedding/HE/pt_files'
pt_output_dir = '/local/data2/chrsp39/CBTN_v2/virchow2_embedding/Merged_HE/pt_files'

if not os.path.exists(pt_output_dir):
    os.makedirs(pt_output_dir)

tensor_files = os.listdir(pt_files)

subjects_id = []

for tensor_file in tensor_files:
    subject_id = tensor_file.split('___')[0]
    subjects_id.append(subject_id)

subjects_id = set(subjects_id)

counter = 1

for subject_id in subjects_id:
    stacked_tensors = []
    tensors_to_stack = []

    for tensor_file in tensor_files:
        if tensor_file.startswith(subject_id + '___'):
            tensor_path = os.path.join(pt_files, tensor_file)
            tensor = torch.load(tensor_path)
            tensors_to_stack.append(tensor)

    stacked_tensor = torch.cat(tensors_to_stack, dim=0)
    output_file_path = os.path.join(pt_output_dir, f"{subject_id}.pt")
    torch.save(stacked_tensor, output_file_path)

    stacked_tensors = []
    tensors_to_stack = []
    print(f'{counter}/{len(subjects_id)} subjects stacked', end='\r')
    counter += 1

print('\nMerging finished!')

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

# %% STACK DOUBLE MODALITY MATCHED SUBJECTS
first_modality_pt_files = '/local/data2/chrsp39/CBTN_v2/virchow2_class_token/Merged_HE/pt_files'
second_modality_pt_files = '/local/data2/chrsp39/CBTN_v2/virchow2_class_token/Merged_GFAP/pt_files'
pt_output_dir = '/local/data2/chrsp39/CBTN_v2/virchow2_class_token/Merged_HE_GFAP/pt_files'

if not os.path.exists(pt_output_dir):
    os.makedirs(pt_output_dir)

first_modality_subjects_id = {file.split('.')[0] for file in os.listdir(first_modality_pt_files)}
first_modality_subjects_id = set(first_modality_subjects_id)

second_modality_subjects_id = {file.split('.')[0] for file in os.listdir(second_modality_pt_files)}
second_modality_subjects_id = set(second_modality_subjects_id)

common_subjects = first_modality_subjects_id & second_modality_subjects_id

counter = 1

for subject_id in common_subjects:
    second_modality_subject_id_tensor_path = os.path.join(second_modality_pt_files, subject_id + '.pt')
    second_modality_subject_id_tensor = torch.load(second_modality_subject_id_tensor_path)

    first_modality_subject_id_tensor_path = os.path.join(first_modality_pt_files, subject_id + '.pt')
    first_modality_subject_id_tensor = torch.load(first_modality_subject_id_tensor_path)

    stacked_tensor = torch.cat([second_modality_subject_id_tensor, first_modality_subject_id_tensor], dim=0)
    output_file_path = os.path.join(pt_output_dir, f"{subject_id}.pt")
    torch.save(stacked_tensor, output_file_path)

    print(f'{counter}/{len(common_subjects)} subjects stacked', end='\r')
    counter += 1

print('\nMerging finished!')

# %% STACK TRIPLE MODALITY MATCHED SUBJECTS
first_modality_pt_files = '/local/data2/chrsp39/CBTN_v2/virchow2_class_token/Merged_HE/pt_files'
second_modality_pt_files = '/local/data2/chrsp39/CBTN_v2/virchow2_class_token/Merged_KI67/pt_files'
third_modality_pt_files = '/local/data2/chrsp39/CBTN_v2/virchow2_class_token/Merged_GFAP/pt_files'
pt_output_dir = '/local/data2/chrsp39/CBTN_v2/virchow2_class_token/Merged_HE_KI67_GFAP/pt_files'

if not os.path.exists(pt_output_dir):
    os.makedirs(pt_output_dir)

first_modality_subjects_id = {file.split('.')[0] for file in os.listdir(first_modality_pt_files)}
second_modality_subjects_id = {file.split('.')[0] for file in os.listdir(second_modality_pt_files)}
third_modality_subjects_id = {file.split('.')[0] for file in os.listdir(third_modality_pt_files)}

common_subjects = first_modality_subjects_id & second_modality_subjects_id & third_modality_subjects_id

counter = 1

for subject_id in common_subjects:
    first_modality_tensor_path = os.path.join(first_modality_pt_files, f"{subject_id}.pt")
    second_modality_tensor_path = os.path.join(second_modality_pt_files, f"{subject_id}.pt")
    third_modality_tensor_path = os.path.join(third_modality_pt_files, f"{subject_id}.pt")

    first_modality_tensor = torch.load(first_modality_tensor_path)
    second_modality_tensor = torch.load(second_modality_tensor_path)
    third_modality_tensor = torch.load(third_modality_tensor_path)

    stacked_tensor = torch.cat([first_modality_tensor, second_modality_tensor, third_modality_tensor], dim=0)

    output_file_path = os.path.join(pt_output_dir, f"{subject_id}.pt")
    torch.save(stacked_tensor, output_file_path)

    print(f'{counter}/{len(common_subjects)} subjects stacked', end='\r')
    counter += 1

print('\nMerging finished!')

# %% STACK ALL HE & KI67 SUBJECTS
# HE_pt_files = '/local/data2/chrsp39/CBTN_v2/HIPT/Merged_HE/4096_features/slide_features'
# KI67_pt_files = '/local/data2/chrsp39/CBTN_v2/HIPT/Merged_KI67/4096_features/slide_features'
# HE_KI67_pt_files = '/local/data2/chrsp39/CBTN_v2/HIPT/Merged_Histology/4096_bounded_features/slide_features'
# pt_output_dir = '/local/data2/chrsp39/CBTN_v2/HIPT/Merged_Histology/4096_features/slide_features'

# if not os.path.exists(pt_output_dir):
#     os.makedirs(pt_output_dir)

# HE_KI67_tensor_files = os.listdir(HE_KI67_pt_files)
# counter = 1
# for HE_KI67_tensor_file in HE_KI67_tensor_files: 
#     HE_KI67_tensor_path = os.path.join(HE_KI67_pt_files, HE_KI67_tensor_file)
#     shutil.copyfile(HE_KI67_tensor_path, os.path.join(pt_output_dir, HE_KI67_tensor_file))
#     print(f'{counter}/{len(HE_KI67_tensor_files)} subjects copyied', end='\r')
#     counter += 1

# HE_tensor_files = os.listdir(HE_pt_files)
# counter = 1
# for HE_tensor_file in HE_tensor_files: 
#     if HE_tensor_file not in HE_KI67_tensor_files:
#         HE_tensor_path = os.path.join(HE_pt_files, HE_tensor_file)
#         shutil.copyfile(HE_tensor_path, os.path.join(pt_output_dir, HE_tensor_file))
#     print(f'{counter}/{len(HE_tensor_files)} subjects copyied', end='\r')
#     counter += 1

# KI67_tensor_files = os.listdir(KI67_pt_files)
# counter = 1
# for KI67_tensor_file in KI67_tensor_files:
#     if KI67_tensor_file not in HE_KI67_tensor_files:
#         KI_67_tensor_path = os.path.join(KI67_pt_files, KI67_tensor_file)
#         shutil.copyfile(KI_67_tensor_path, os.path.join(pt_output_dir, KI67_tensor_file))
#     print(f'{counter}/{len(KI67_tensor_files)} subjects copyied', end='\r')
#     counter += 1
# %%
