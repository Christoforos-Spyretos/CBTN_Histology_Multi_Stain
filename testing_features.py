# %% IMPORTS
import torch

# %% LOAD PATHS
subject_level_attention = '/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_ATTENTION_HE/LGG_vs_HGG/features/conch_v1_5/train/pt_files/fold_0/C17097.pt'
cross_attented_feature = '/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_CROSS_ATTENTION_HE_KI67/LGG_vs_HGG/features/conch_v1_5/train/fold_0/pt_files/C17097.pt'

# %% LOAD AND INSPECT .pt FILES
def print_pt_info(path):
	data = torch.load(path, map_location='cpu', weights_only=False)
	print(f"File: {path}")
	if isinstance(data, dict):
		print("Type: dict")
		print("Keys:", list(data.keys()))
		for k, v in data.items():
			if hasattr(v, 'shape'):
				print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
			else:
				print(f"  {k}: type={type(v)}")
	elif hasattr(data, 'shape'):
		print(f"Type: {type(data)}, shape={data.shape}, dtype={data.dtype}")
	else:
		print(f"Type: {type(data)}")
	print("-" * 40)

# %%
print_pt_info(subject_level_attention)
print_pt_info(cross_attented_feature)

# %%
# imports
import torch
import os
import numpy as np


path_to_feature = '/local/data3/chrsp39/CBTN_v2/SUBJECT_LEVEL_CROSS_ATTENTION_HE_KI67/LGG_vs_HGG/features/conch_v1_5/train/fold_0/pt_files/C15252.pt'

print_pt_info(path_to_feature)


# %%
path_to_feature = '/local/data3/chrsp39/CBTN_v2/Merged_KI67/features/conch_v1_5/pt_files/C15252.pt'

print_pt_info(path_to_feature)

# %%