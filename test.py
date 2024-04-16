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
pt_file = torch.load("/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE/merged_vit_features/pt_files/C15252.pt")
print(pt_file.shape)
print(pt_file)

# %%

pt_file = torch.load("/local/data2/chrsp39/CBTN_v2/CLAM/Merged_KI67/merged_vit_features/pt_files/C15252.pt")
print(pt_file.shape)
print(pt_file)

# %%
pt_file = torch.load("/local/data2/chrsp39/CBTN_v2/CLAM/Merged_HE_KI67/merged_vit_features/pt_files/C15252.pt")
print(pt_file.shape)
print(pt_file)
# %%
