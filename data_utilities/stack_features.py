# %% IMPORTS
import h5py
import torch
import os
import numpy as np
import shutil
import argparse
import yaml

# LOAD CONFIG
parser = argparse.ArgumentParser(description='Stack feature tensors across modalities')
parser.add_argument('--config', type=str, 
                    default='configs/data_utilities/stack_features.yaml',
                    help='Path to config file')
args = parser.parse_args()

# Load configuration from YAML file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

print(f"Configuration loaded from: {args.config}")
print("="*60)

# STACK SINGLE MODALITY
if config['stack_single_modality']:
    print("Starting STACK SINGLE MODALITY...")
    first_modality_pt_files = config['first_modality_pt_files']
    pt_output_dir = config['saved_stack_single_modality']

    if first_modality_pt_files is None:
        print("Error: first_modality_pt_files is None. Cannot stack single modality.")
    elif pt_output_dir is None:
        print("Error: saved_stack_single_modality is None. Cannot stack single modality.")
    else:
        if not os.path.exists(pt_output_dir):
            os.makedirs(pt_output_dir)

tensor_files = os.listdir(first_modality_pt_files)

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
            tensor_path = os.path.join(first_modality_pt_files, tensor_file)
            tensor = torch.load(tensor_path)
            tensors_to_stack.append(tensor)

    stacked_tensor = torch.cat(tensors_to_stack, dim=0)
    output_file_path = os.path.join(pt_output_dir, f"{subject_id}.pt")
    torch.save(stacked_tensor, output_file_path)

    stacked_tensors = []
    tensors_to_stack = []
    print(f'{counter}/{len(subjects_id)} subjects stacked', end='\r')
    counter += 1

    print('\nSingle modality stacking finished!')
else:
    print("STACK SINGLE MODALITY: Skipped (disabled in config)")

# STACK DOUBLE MODALITY MATCHED SUBJECTS
if config['stack_double_modality']:
    print("Starting STACK DOUBLE MODALITY MATCHED SUBJECTS...")
    first_modality_pt_files = config['first_modality_pt_files']
    second_modality_pt_files = config['second_modality_pt_files']
    pt_output_dir = config['saved_stack_double_modality']

    if first_modality_pt_files is None or second_modality_pt_files is None:
        print("Error: first_modality_pt_files or second_modality_pt_files is None. Cannot stack double modality.")
    elif pt_output_dir is None:
        print("Error: saved_stack_double_modality is None. Cannot stack double modality.")
    else:
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

        print('\nDouble modality stacking finished!')
else:
    print("STACK DOUBLE MODALITY: Skipped (disabled in config)")

# STACK TRIPLE MODALITY MATCHED SUBJECTS
if config['stack_triple_modality']:
    print("Starting STACK TRIPLE MODALITY MATCHED SUBJECTS...")
    first_modality_pt_files = config['first_modality_pt_files']
    second_modality_pt_files = config['second_modality_pt_files']
    third_modality_pt_files = config['third_modality_pt_files']
    pt_output_dir = config['saved_stack_triple_modality']
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

    print('\nTriple modality stacking finished!')
else:
    print("STACK TRIPLE MODALITY: Skipped (disabled in config)")

print("All stacking operations completed!")

