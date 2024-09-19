# imports
import time
import os
import argparse
import pdb
from functools import partial
import torch
import torch.nn as nn
import timm
import glob
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
import yaml
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
from tqdm import tqdm
import numpy as np

# pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# internal imports
from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
		# if you are using multi-GPU.
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_w_loader(output_path, loader, model, verbose = 0):
	"""
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

def build_experiment_name(cfg):	
	return '_'.join([str(cfg.seed),
					cfg.data_h5_dir, 
					cfg.data_slide_dir, 
					cfg.slide_ext, 
					cfg.csv_path, 
					cfg.feat_dir, 
					cfg.model_name, 
					str(cfg.batch_size), 
					str(cfg.num_workers),
					str(cfg.no_auto_skip), 
					str(cfg.target_patch_size)])

@hydra.main(version_base="1.3.2", 
			config_path= '/home/chrsp39/CBTN_Histology_Multi_Modal/configs/pre_processing', 
			config_name= 'extract_features_fp')
			
def main(cfg:DictConfig):

	for data_h5_dir, data_slide_dir, csv_path, feat_dir in zip(cfg.data_h5_dirs, cfg.data_slide_dirs, cfg.csv_paths, cfg.feat_dirs):
		# Update the configuration for each run
		with open_dict(cfg):
			cfg.data_h5_dir = data_h5_dir
			cfg.data_slide_dir = data_slide_dir
			cfg.csv_path = csv_path
			cfg.feat_dir = feat_dir

		# seed everything
		seed_torch(cfg.seed)
		print(f'Processing {cfg.data_h5_dir} with slide directory {cfg.data_slide_dir}')
	
		settings ={'seed' : cfg.seed,
			'data_h5_dir' : cfg.data_h5_dir,
			'data_slide_dir' : cfg.data_slide_dir,
			'slide_ext' : cfg.slide_ext,
			'csv_path' : cfg.csv_path,
			'feat_dir' : cfg.feat_dir,
			'model_name' : cfg.model_name,
			'batch_size' : cfg.batch_size,
			'num_workers' : cfg.num_workers,
			'no_auto_skip' : cfg.no_auto_skip,
			'target_patch_size' : cfg.target_patch_size}

		# get experiment name 
		print(cfg)
		experiment_name = build_experiment_name(cfg)

		print('initializing dataset')
		csv_path = cfg.csv_path
		if csv_path is None:
			raise NotImplementedError("CSV path must be provided in the configuration.")

		bags_dataset = Dataset_All_Bags(csv_path)
		
		os.makedirs(cfg.feat_dir, exist_ok=True)
		os.makedirs(os.path.join(cfg.feat_dir, 'pt_files'), exist_ok=True)
		os.makedirs(os.path.join(cfg.feat_dir, 'h5_files'), exist_ok=True)
		dest_files = os.listdir(os.path.join(cfg.feat_dir, 'pt_files'))

		model, img_transforms = get_encoder(cfg.model_name, target_img_size=cfg.target_patch_size)
				
		_ = model.eval()
		model = model.to(device)
		total = len(bags_dataset)

		loader_kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True} if device.type == "cuda" else {}

		for bag_candidate_idx in tqdm(range(total)):
			slide_id = bags_dataset[bag_candidate_idx].split(cfg.slide_ext)[0]
			bag_name = slide_id+'.h5'
			h5_file_path = os.path.join(cfg.data_h5_dir, 'patches', bag_name)
			slide_file_path = os.path.join(cfg.data_slide_dir, slide_id+cfg.slide_ext)
			print('\nprogress: {}/{}'.format(bag_candidate_idx+1, total))
			print(slide_id)

			if not cfg.no_auto_skip and slide_id+'.pt' in dest_files:
				print('skipped {}'.format(slide_id))
				continue 

			output_path = os.path.join(cfg.feat_dir, 'h5_files', bag_name)
			time_start = time.time()
			wsi = openslide.open_slide(slide_file_path)
			dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
										wsi=wsi, 
										img_transforms=img_transforms)

			loader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, **loader_kwargs)
			output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 1)

			time_elapsed = time.time() - time_start
			print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

			with h5py.File(output_file_path, "r") as file:
				features = file['features'][:]
				print('features size: ', features.shape)
				print('coordinates size: ', file['coords'].shape)

			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name)
			torch.save(features, os.path.join(cfg.feat_dir, 'pt_files', bag_base+'.pt'))

if __name__ == '__main__':
	main()
	print("finished!")