# %% IMPORTS
from PIL import Image, ImageOps
import numpy as np
from scipy.stats import spearmanr
import os
from glob import glob

Image.MAX_IMAGE_PIXELS = 1000000000

# %% LOAD PATHS
path_to_LGG_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/5_class/KI67/LGG'
path_to_HGG_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/5_class/KI67/HGG'
path_to_EP_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/5_class/KI67/EP'
path_to_MB_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/5_class/KI67/MB'
path_to_GG_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/5_class/KI67/GG'
path_to_density_and_KI67_LI_maps = '/local/data1/chrsp39/QuPath_Portable/CBTN_Results/Normalized_Density_Maps_2'

# Helper to extract slide_id from heatmap filename
def extract_slide_id(heatmap_path):
    basename = os.path.basename(heatmap_path)
    parts = basename.split('_0.5_roi_')[0]
    return parts

# Helper to get LI map path from slide_id
def get_li_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_Ki67_LI_map.png')

# Helper to get negative density map path from slide_id
def get_negative_density_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_NegDMap.png')

# Helper to get positive density map path from slide_id
def get_positive_density_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_PosDMap.png')

# Collect heatmap paths per label (filter by suffix)
heatmap_suffix = '_0.5_roi_0_blur_1_rs_0_bc_1_a_1.0_l_1_bi_0_-1.0.jpg'
label_paths = {
    'LGG': [f for f in glob(os.path.join(path_to_LGG_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
    'HGG': [f for f in glob(os.path.join(path_to_HGG_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
    'MB':  [f for f in glob(os.path.join(path_to_MB_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
    'EP':  [f for f in glob(os.path.join(path_to_EP_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
    'GG':  [f for f in glob(os.path.join(path_to_GG_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
}

# %% CORRELATION ANALYSIS BETWEEN HEATMAPS AND LI MAPS
print("Correlation analysis between heatmaps and KI67 LI maps:")

# Store correlations
results_LI_maps = {}
all_heatmap_fg_raw = []
all_li_map_fg_raw = []

for label, heatmap_paths in label_paths.items():
    label_heatmap_fg_raw = []
    label_li_map_fg_raw = []
    for heatmap_path in heatmap_paths:
        slide_id = extract_slide_id(heatmap_path)
        li_map_path = get_li_map_path(slide_id)
        if not os.path.exists(li_map_path):
            print(f"LI map not found for {slide_id}")
            continue
        # load heatmap
        heatmap_im = ImageOps.grayscale(Image.open(heatmap_path))
        heatmap_array = np.array(heatmap_im).astype(float)
        # load KI67 LI map
        li_map_im = Image.open(li_map_path)
        if li_map_im.mode != 'L':
            li_map_im = li_map_im.convert('L')
        li_width, li_height = li_map_im.size
        # resize heatmap to match LI map dimensions
        heatmap_resized = heatmap_im.resize((li_width, li_height))
        heatmap_resized_array = np.array(heatmap_resized).astype(float)
        li_map_array = np.array(li_map_im).astype(float)
        # extract foreground pixels only 
        heatmap_bg_mask = heatmap_resized_array <= 5
        li_map_bg_mask = li_map_array == 0
        combined_bg_mask = heatmap_bg_mask | li_map_bg_mask
        heatmap_fg_raw = heatmap_resized_array[~combined_bg_mask]
        li_map_fg_raw = li_map_array[~combined_bg_mask]
        label_heatmap_fg_raw.extend(heatmap_fg_raw)
        label_li_map_fg_raw.extend(li_map_fg_raw)
        all_heatmap_fg_raw.extend(heatmap_fg_raw)
        all_li_map_fg_raw.extend(li_map_fg_raw)

    if len(label_heatmap_fg_raw) > 0:
        corr_raw, pval_raw = spearmanr(label_heatmap_fg_raw, label_li_map_fg_raw)
        print(f"[{label}] Spearman correlation: {corr_raw:.3f}")
        results_LI_maps[label] = {'raw_corr': corr_raw}
    else:
        print(f"[{label}] No foreground pixels found for correlation")
        results_LI_maps[label] = {'raw_corr': None}

if len(all_heatmap_fg_raw) > 0:
    corr_raw_total, pval_raw_total = spearmanr(all_heatmap_fg_raw, all_li_map_fg_raw)
    print(f"[TOTAL] Spearman correlation: {corr_raw_total:.3f}")
else:
    print("[TOTAL] No foreground pixels found for correlation")

# %% CORRELATION ANALYSIS BETWEEN HEATMAPS AND NEGATIVE DENSITY MAPS
print("Correlation analysis between heatmaps and negative density maps:")

results_negative_density_maps = {}
all_heatmap_fg_raw = []
all_neg_density_fg_raw = []

for label, heatmap_paths in label_paths.items():
    label_heatmap_fg_raw = []
    label_neg_density_fg_raw = []
    for heatmap_path in heatmap_paths:
        slide_id = extract_slide_id(heatmap_path)
        neg_density_map_path = get_negative_density_map_path(slide_id)
        if not os.path.exists(neg_density_map_path):
            print(f"Negative density map not found for {slide_id}")
            continue
        # load heatmap
        heatmap_im = ImageOps.grayscale(Image.open(heatmap_path))
        heatmap_array = np.array(heatmap_im).astype(float)
        # load negative density map
        neg_density_map_im = Image.open(neg_density_map_path)
        if neg_density_map_im.mode != 'L':
            neg_density_map_im = neg_density_map_im.convert('L')
        neg_density_width, neg_density_height = neg_density_map_im.size
        # resize heatmap to match negative density map dimensions
        heatmap_resized = heatmap_im.resize((neg_density_width, neg_density_height))
        heatmap_resized_array = np.array(heatmap_resized).astype(float)
        neg_density_map_array = np.array(neg_density_map_im).astype(float)
        # extract foreground pixels only 
        heatmap_bg_mask = heatmap_resized_array <= 5
        neg_density_map_bg_mask = neg_density_map_array == 0
        combined_bg_mask = heatmap_bg_mask | neg_density_map_bg_mask
        heatmap_fg_raw = heatmap_resized_array[~combined_bg_mask]
        neg_density_map_fg_raw = neg_density_map_array[~combined_bg_mask]
        label_heatmap_fg_raw.extend(heatmap_fg_raw)
        label_neg_density_fg_raw.extend(neg_density_map_fg_raw)
        all_heatmap_fg_raw.extend(heatmap_fg_raw)
        all_neg_density_fg_raw.extend(neg_density_map_fg_raw)

    if len(label_heatmap_fg_raw) > 0:
        corr_raw, pval_raw = spearmanr(label_heatmap_fg_raw, label_neg_density_fg_raw)
        print(f"[{label}] Spearman correlation: {corr_raw:.3f}")
        results_negative_density_maps[label] = {'raw_corr': corr_raw}
    else:
        print(f"[{label}] No foreground pixels found for correlation")
        results_negative_density_maps[label] = {'raw_corr': None}

if len(all_heatmap_fg_raw) > 0:
    corr_raw_total, pval_raw_total = spearmanr(all_heatmap_fg_raw, all_neg_density_fg_raw)
    print(f"[TOTAL] Spearman correlation: {corr_raw_total:.3f}")
else:
    print("[TOTAL] No foreground pixels found for correlation")

# %% CORRELATION ANALYSIS BETWEEN HEATMAPS AND POSITIVE DENSITY MAPS
print("Correlation analysis between heatmaps and positive density maps:")

results_positive_density_maps = {}
all_heatmap_fg_raw = []
all_pos_density_fg_raw = []

for label, heatmap_paths in label_paths.items():
    label_heatmap_fg_raw = []
    label_pos_density_fg_raw = []
    for heatmap_path in heatmap_paths:
        slide_id = extract_slide_id(heatmap_path)
        pos_density_map_path = get_positive_density_map_path(slide_id)
        if not os.path.exists(pos_density_map_path):
            print(f"Positive density map not found for {slide_id}")
            continue
        # load heatmap
        heatmap_im = ImageOps.grayscale(Image.open(heatmap_path))
        heatmap_array = np.array(heatmap_im).astype(float)
        # load positive density map
        pos_density_map_im = Image.open(pos_density_map_path)
        if pos_density_map_im.mode != 'L':
            pos_density_map_im = pos_density_map_im.convert('L')
        pos_density_width, pos_density_height = pos_density_map_im.size
        # resize heatmap to match positive density map dimensions
        heatmap_resized = heatmap_im.resize((pos_density_width, pos_density_height))
        heatmap_resized_array = np.array(heatmap_resized).astype(float)
        pos_density_map_array = np.array(pos_density_map_im).astype(float)
        # extract foreground pixels only 
        heatmap_bg_mask = heatmap_resized_array <= 5
        pos_density_map_bg_mask = pos_density_map_array == 0
        combined_bg_mask = heatmap_bg_mask | pos_density_map_bg_mask
        heatmap_fg_raw = heatmap_resized_array[~combined_bg_mask]
        pos_density_map_fg_raw = pos_density_map_array[~combined_bg_mask]
        label_heatmap_fg_raw.extend(heatmap_fg_raw)
        label_pos_density_fg_raw.extend(pos_density_map_fg_raw)
        all_heatmap_fg_raw.extend(heatmap_fg_raw)
        all_pos_density_fg_raw.extend(pos_density_map_fg_raw)

    if len(label_heatmap_fg_raw) > 0:
        corr_raw, pval_raw = spearmanr(label_heatmap_fg_raw, label_pos_density_fg_raw)
        print(f"[{label}] Spearman correlation: {corr_raw:.3f}")
        results_positive_density_maps[label] = {'raw_corr': corr_raw}
    else:
        print(f"[{label}] No foreground pixels found for correlation")
        results_positive_density_maps[label] = {'raw_corr': None}

if len(all_heatmap_fg_raw) > 0:
    corr_raw_total, pval_raw_total = spearmanr(all_heatmap_fg_raw, all_pos_density_fg_raw)
    print(f"[TOTAL] Spearman correlation: {corr_raw_total:.3f}")
else:
    print("[TOTAL] No foreground pixels found for correlation")

# %%