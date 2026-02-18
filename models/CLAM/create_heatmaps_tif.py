from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import warnings
import h5py

from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models import get_encoder
from types import SimpleNamespace
from collections import namedtuple
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils_tif import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5, open_hdf5_with_retry
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
    features = features.to(device)
    with torch.inference_mode():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            model_results_dict = model(features)
            logits, Y_prob, Y_hat, A, _, _ = model(features)
            Y_hat = Y_hat.item()

            if isinstance(model, (CLAM_MB,)):
                A = A[Y_hat]

            A = A.view(-1, 1).cpu().numpy()

        else:
            raise NotImplementedError

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
        
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params

@hydra.main(version_base="1.3.2",
            config_path='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/configs/heatmaps',
            config_name='create_heatmaps')
def main(cfg: DictConfig):
    
    seed_torch(cfg.seed)
    
    # Save configuration to text file in experiment directory
    os.makedirs(cfg.exp_arguments.save_exp_code, exist_ok=True)
    
    config_save_path = os.path.join(cfg.exp_arguments.save_exp_code, 'heatmap_config_tif.txt')
    with open(config_save_path, 'w') as f:
        f.write("Heatmap generation configuration (TIF):\n")
        f.write(f"Task: {cfg.exp_arguments.task}\n")
        f.write(f"Save Code: {cfg.exp_arguments.save_exp_code}\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        f.write("All configuration parameters:\n")
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Configuration saved to: {config_save_path}")
    
    # Print configuration summary
    print("="*80)
    print(f"Heatmap Generation (TIF) - Task: {cfg.exp_arguments.task}")
    print("="*80)
    print("\nConfiguration Summary:")
    for key in cfg.keys():
        if key != '_target_':
            print(f"{key}:")
            if OmegaConf.is_dict(cfg[key]):
                for subkey, subval in cfg[key].items():
                    print(f"  {subkey}: {subval}")
            else:
                print(f"  {cfg[key]}")
    
    patch_size = tuple([cfg.patching_arguments.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - cfg.patching_arguments.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], cfg.patching_arguments.overlap, step_size[0], step_size[1]))

    preset = cfg.data_arguments.preset
    # Load default parameters from config
    def_seg_params = OmegaConf.to_container(cfg.seg_params, resolve=True)
    def_filter_params = OmegaConf.to_container(cfg.filter_params, resolve=True)
    def_vis_params = OmegaConf.to_container(cfg.vis_params, resolve=True)
    def_patch_params = OmegaConf.to_container(cfg.patch_params, resolve=True)

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            if key in preset_df.columns:
                def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            if key in preset_df.columns:
                def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            if key in preset_df.columns:
                def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            if key in preset_df.columns:
                def_patch_params[key] = preset_df.loc[0, key]


    if cfg.data_arguments.process_list is None:
        if isinstance(cfg.data_arguments.data_dir, list):
            slides = []
            for data_dir in cfg.data_arguments.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(cfg.data_arguments.data_dir))
        slides = [slide for slide in slides if cfg.data_arguments.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
        
    else:
        df = pd.read_csv(cfg.data_arguments.process_list)
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = cfg.model_arguments.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))
    
    # Create model args namespace
    model_args = SimpleNamespace(
        ckpt_path=cfg.model_arguments.ckpt_path,
        model_type=cfg.model_arguments.get('model_type', 'clam_sb'),
        model_size=cfg.model_arguments.model_size,
        drop_out=cfg.model_arguments.drop_out,
        embed_dim=cfg.model_arguments.embed_dim,
        n_classes=cfg.exp_arguments.n_classes
    )
    
    if cfg.model_arguments.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError

    feature_extractor, img_transforms = get_encoder(cfg.encoder_arguments.model_name, target_img_size=cfg.encoder_arguments.target_img_size)
    _ = feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    print('Done!')

    label_dict = cfg.data_arguments.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
    

    # os.makedirs(cfg.exp_arguments.production_save_dir, exist_ok=True)
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
    'custom_downsample': cfg.patching_arguments.custom_downsample, 'level': cfg.patching_arguments.patch_level, 'use_center_shift': cfg.heatmap_arguments.use_center_shift}

    for i in tqdm(range(len(process_stack))):
        slide_name = process_stack.loc[i, 'slide_id']
        if cfg.data_arguments.slide_ext not in slide_name:
            slide_name += cfg.data_arguments.slide_ext
        print('\nprocessing: ', slide_name)	

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(cfg.data_arguments.slide_ext, '')

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        p_slide_save_dir = os.path.join(cfg.exp_arguments.save_exp_code, str(grouping), slide_id)
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = p_slide_save_dir

        if cfg.heatmap_arguments.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(cfg.data_arguments.data_dir, str):
            slide_path = os.path.join(cfg.data_arguments.data_dir, slide_name)
        elif isinstance(cfg.data_arguments.data_dir, dict):
            data_dir_key = process_stack.loc[i, cfg.data_arguments.data_dir_key]
            slide_path = os.path.join(cfg.data_arguments.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))
        
        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[cfg.patching_arguments.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * cfg.patching_arguments.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(p_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params)
        
        # Downscale mask if it's too large (to keep file size manageable)
        max_dimension = 3072  # Maximum width or height
        if max(mask.size) > max_dimension:
            scale_factor = max_dimension / max(mask.size)
            new_size = (int(mask.size[0] * scale_factor), int(mask.size[1] * scale_factor))
            mask = mask.resize(new_size, Image.LANCZOS)
            print(f'Downscaled mask to {new_size} to reduce file size')
        
        mask.save(mask_path, quality=85)
        
        features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
    

        ##### check if h5_features_file exists ######
        if not os.path.isfile(h5_path) :
            _, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
                                            model=model, 
                                            feature_extractor=feature_extractor, 
                                            img_transforms=img_transforms,
                                            batch_size=cfg.exp_arguments.batch_size, **blocky_wsi_kwargs, 
                                            attn_save_path=None, feat_save_path=h5_path, 
                                            ref_scores=None)				
        
        ##### check if pt_features_file exists ######
        if not os.path.isfile(features_path):
            file = open_hdf5_with_retry(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            file.close()

        # load features 
        features = torch.load(features_path)
        process_stack.loc[i, 'bag_size'] = len(features)
        
        wsi_object.saveSegmentation(mask_file)
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, cfg.exp_arguments.n_classes)
        del features
        
        if not os.path.isfile(block_map_save_path): 
            file = open_hdf5_with_retry(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        
        # save top 3 predictions
        for c in range(cfg.exp_arguments.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

        if cfg.data_arguments.process_list is not None:
            save_path = cfg.data_arguments.process_list.replace('.csv', '') + '.csv'
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            process_stack.to_csv(save_path, index=False)
        else:
            save_path = 'heatmaps/results/{}.csv'.format(cfg.exp_arguments.save_exp_code)
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            process_stack.to_csv(save_path, index=False)
        
        file = open_hdf5_with_retry(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        samples = cfg.sample_arguments.samples
        for sample in samples:
            if sample['sample']:
                tag = "label_{}_pred_{}".format(label, Y_hats[0])
                sample_save_dir = os.path.join(cfg.exp_arguments.save_exp_code, str(grouping), slide_id, 'sampled_patches', sample['name'])
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                    score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), cfg.patching_arguments.patch_level, (cfg.patching_arguments.patch_size, cfg.patching_arguments.patch_size)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
        'custom_downsample': cfg.patching_arguments.custom_downsample, 'level': cfg.patching_arguments.patch_level, 'use_center_shift': cfg.heatmap_arguments.use_center_shift}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            pass
        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=cfg.heatmap_arguments.cmap, alpha=cfg.heatmap_arguments.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                            thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
        
            heatmap.save(os.path.join(p_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, cfg.patching_arguments.overlap, cfg.heatmap_arguments.use_roi))

        if cfg.heatmap_arguments.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None
        
        if cfg.heatmap_arguments.calc_heatmap:
            compute_from_patches(wsi_object=wsi_object, 
                                img_transforms=img_transforms,
                                clam_pred=Y_hats[0], model=model, 
                                feature_extractor=feature_extractor, 
                                batch_size=cfg.exp_arguments.batch_size, **wsi_kwargs, 
                                attn_save_path=save_path,  ref_scores=ref_scores)

        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if cfg.heatmap_arguments.use_roi:
                save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, cfg.patching_arguments.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue
        
        with open_hdf5_with_retry(save_path, 'r') as file:
            dset = file['attention_scores']
            coord_dset = file['coords']
            scores = dset[:]
            coords = coord_dset[:]

        heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': cfg.heatmap_arguments.vis_level, 'blur': cfg.heatmap_arguments.blur, 'custom_downsample': cfg.heatmap_arguments.custom_downsample}
        if cfg.heatmap_arguments.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(cfg.patching_arguments.overlap), int(cfg.heatmap_arguments.use_roi),
                                                                                        int(cfg.heatmap_arguments.blur), 
                                                                                        int(cfg.heatmap_arguments.use_ref_scores), int(cfg.heatmap_arguments.blank_canvas), 
                                                                                        float(cfg.heatmap_arguments.alpha), int(cfg.heatmap_arguments.vis_level), 
                                                                                        int(cfg.heatmap_arguments.binarize), float(cfg.heatmap_arguments.binary_thresh), cfg.heatmap_arguments.save_ext)


        if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            pass
        
        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
                                  cmap=cfg.heatmap_arguments.cmap, alpha=cfg.heatmap_arguments.alpha, **heatmap_vis_args, 
                                  binarize=cfg.heatmap_arguments.binarize, 
                                    blank_canvas=cfg.heatmap_arguments.blank_canvas,
                                    thresh=cfg.heatmap_arguments.binary_thresh,  patch_size = vis_patch_size,
                                    overlap=cfg.patching_arguments.overlap, 
                                    top_left=top_left, bot_right = bot_right)
            if cfg.heatmap_arguments.save_ext == 'jpg':
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
            else:
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
        
        if cfg.heatmap_arguments.save_orig:
            if cfg.heatmap_arguments.vis_level >= 0:
                vis_level = cfg.heatmap_arguments.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id, int(vis_level), cfg.heatmap_arguments.save_ext)
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=cfg.heatmap_arguments.custom_downsample)
                if cfg.heatmap_arguments.save_ext == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

    # Save final config as YAML
    final_config_path = os.path.join(cfg.exp_arguments.save_exp_code, 'config.yaml')
    with open(final_config_path, 'w') as outfile:
        outfile.write(OmegaConf.to_yaml(cfg))
    
    print(f"\nHeatmap generation complete!")
    print(f"Configuration saved to: {final_config_path}")

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")
