# imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm
import yaml
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
from PIL import Image

# internal imports
from wsi_core.WholeSlideImage_tif import WholeSlideImage
from wsi_core.wsi_utils_tif import StitchCoords
from wsi_core.batch_process_utils import initialize_df

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	
	bg_color = (0, 0, 0) 
	
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=bg_color, alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none', 'ref_patch_size': 512},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None):
	


	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in tqdm(range(total)):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i+1/total, i+1, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = os.path.join(source, slide)
		try:
			WSI_object = WholeSlideImage(full_path)
		except Exception as e:
			print(f'Failed to initialize WSI for {slide}: {str(e)}')
			df.loc[idx, 'status'] = 'failed_wsi_init'
			continue

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			# Use higher level (lower resolution) for TIFF to reduce mask file size
			# Level 2 corresponds to 16x downsampling, similar to SVS behavior
			current_vis_params['vis_level'] = 2

		if current_seg_params['seg_level'] < 0:
			current_seg_params['seg_level'] = current_seg_params['seg_level']

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []
		
		# Check if image dimensions are too large for segmentation
		w, h = WSI_object.level_dim[current_seg_params['seg_level']]
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue


		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.png')  # Changed to PNG for lossless quality
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

def build_experiment_name(cfg):
	return '_'.join[(cfg.source,
				  cfg.save_dir,
				  
				  cfg.patch,
				  cfg.seg,
				  cfg.stitch,
				  cfg.save_mask,
				  cfg.no_auto_skip,
				  cfg.preset,
				  cfg.process_list,
				  cfg.use_default_params,
				  
				  str(cfg.step_size),
				  str(cfg.patch_size),
				  cfg.use_padding,
				  cfg.contour_fn,

				  str(cfg.patch_level),
				  str(cfg.seg_level),
				  str(cfg.sthresh),
				  str(cfg.mthresh),
				  str(cfg.close),
				  cfg.use_otsu,
				  cfg.keep_ids,
				  cfg.exclude_ids,
				  str(cfg.ref_patch_size),

				  str(cfg.a_t),
				  str(cfg.a_h),
				  str(cfg.max_n_holes),

				  str(cfg.vis_level),
				  str(cfg.line_thickness)
				  )]

@hydra.main(version_base="1.3.2", 
			config_path= '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/pre_processing', 
			config_name= 'create_patches_fp')

def main(cfg:DictConfig):

	patch_save_dir = os.path.join(cfg.save_dir, 'patches')
	mask_save_dir = os.path.join(cfg.save_dir, 'masks')
	stitch_save_dir = os.path.join(cfg.save_dir, 'stitches')

	if cfg.process_list:
		process_list = os.path.join(cfg.save_dir, 'process_list')

	else:
		process_list = None

	print('source: ', cfg.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)

	directories = {'source': cfg.source,
				   'save_dir': cfg.save_dir,
				   'patch_save_dir': patch_save_dir,
				   'mask_save_dir': mask_save_dir,
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': cfg.seg_level,
			'sthresh': cfg.sthresh,
			'mthresh': cfg.mthresh,
			'close': cfg.close,
			'use_otsu': cfg.use_otsu,
			'keep_ids': cfg.keep_ids,
			'exclude_ids': cfg.exclude_ids,
			'ref_patch_size': cfg.ref_patch_size}
	
	filter_params = {'a_t': cfg.a_t,
				'a_h': cfg.a_h,
				'max_n_holes': cfg.max_n_holes}

	vis_params = {'vis_level': cfg.vis_level,
			'line_thickness': cfg.line_thickness}

	patch_params = {'use_padding': cfg.use_padding,
				'contour_fn': cfg.contour_fn}

	if cfg.preset:
		preset_df = pd.read_csv(os.path.join(cfg.save_dir,'presets',))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
			'filter_params': filter_params,
			'patch_params': patch_params,
			'vis_params': vis_params}

	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories,
										**parameters,

										patch=cfg.patch,
										seg=cfg.seg,
										stitch=cfg.stitch,
										save_mask=cfg.save_mask,
										auto_skip=cfg.no_auto_skip,
										process_list=process_list,
										use_default_params=cfg.use_default_params,
										
										patch_size = cfg.patch_size, 
										step_size=cfg.step_size, 
										patch_level=cfg.patch_level,
										)

if __name__ == '__main__':
	main()
	print("finished!")