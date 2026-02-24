# %% IMPORTS
from PIL import Image, ImageOps
import numpy as np
from scipy.stats import spearmanr
import os
from glob import glob

Image.MAX_IMAGE_PIXELS = 1000000000

# %% LOAD PATHS
path_to_LGG_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/LGG_vs_HGG/KI67/LGG'
path_to_HGG_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/LGG_vs_HGG/KI67/HGG'
path_to_KI67_LI_maps = '/local/data1/chrsp39/QuPath_Portable/CBTN_Results/Normalized_Density_Maps_2'

# Helper to extract slide_id from heatmap filename
def extract_slide_id(heatmap_path):
    basename = os.path.basename(heatmap_path)
    parts = basename.split('_0.5_roi_')[0]
    return parts

# Helper to get LI map path from slide_id
def get_li_map_path(slide_id):
    return os.path.join(path_to_KI67_LI_maps, f'{slide_id}_Ki67_LI_map.png')

# Collect heatmap paths per label
heatmap_suffix = '_0.5_roi_0_blur_1_rs_0_bc_1_a_1.0_l_1_bi_0_-1.0.jpg'
label_paths = {
    'LGG': [f for f in glob(os.path.join(path_to_LGG_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
    'HGG': [f for f in glob(os.path.join(path_to_HGG_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
}

# Store correlations

import matplotlib.pyplot as plt
results = {}
all_heatmap_fg_raw = []
all_li_map_fg_raw = []
all_heatmap_fg_normd = []
all_li_map_fg_normd = []

for label, heatmap_paths in label_paths.items():
    label_heatmap_fg_raw = []
    label_li_map_fg_raw = []
    label_heatmap_fg_normd = []
    label_li_map_fg_normd = []
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

    # calculate Spearman correlation
    if len(label_heatmap_fg_raw) > 0:
        corr_raw, pval_raw = spearmanr(label_heatmap_fg_raw, label_li_map_fg_raw)
        print(f"[{label}] Spearman correlation: {corr_raw:.3f}")
        results[label] = {'raw_corr': corr_raw}
    else:
        print(f"[{label}] No foreground pixels found for correlation")
        results[label] = {'raw_corr': None}

if len(all_heatmap_fg_raw) > 0:
    corr_raw_total, pval_raw_total = spearmanr(all_heatmap_fg_raw, all_li_map_fg_raw)
    print(f"[TOTAL] Spearman correlation pixels: {corr_raw_total:.3f}")
else:
    print("[TOTAL] No foreground pixels found for correlation")

# %% SINGLE IMAGE CORRELATION
path_to_KI67_heatmap = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/LGG_vs_HGG/KI67/HGG/C54981___7316-277___Ki-67/C54981___7316-277___Ki-67_0.5_roi_0_blur_1_rs_0_bc_1_a_1.0_l_1_bi_0_-1.0.jpg'
path_to_KI67_LI_map = '/local/data1/chrsp39/QuPath_Portable/CBTN_Results/Normalized_Density_Maps_2/C54981___7316-277___Ki-67_Ki67_LI_map.png'

# load heatmap
heatmap_im = ImageOps.grayscale(Image.open(path_to_KI67_heatmap))
heatmap_array = np.array(heatmap_im).astype(float)

# load KI67 LI map
li_map_im = Image.open(path_to_KI67_LI_map)
if li_map_im.mode != 'L':
    li_map_im = li_map_im.convert('L')

li_width, li_height = li_map_im.size

# resize heatmap to match LI map dimensions
heatmap_resized = heatmap_im.resize((li_width, li_height))
heatmap_resized_array = np.array(heatmap_resized).astype(float)

# LI map array
li_map_array = np.array(li_map_im).astype(float)

# extract foreground pixels only 
heatmap_bg_mask = heatmap_resized_array <= 5
li_map_bg_mask = li_map_array == 0
combined_bg_mask = heatmap_bg_mask | li_map_bg_mask
heatmap_fg_raw = heatmap_resized_array[~combined_bg_mask]
li_map_fg_raw = li_map_array[~combined_bg_mask]

# calculate Spearman correlation for pixels
if len(heatmap_fg_raw) > 0:
    corr_raw, pval_raw = spearmanr(heatmap_fg_raw, li_map_fg_raw)
    print(f"Spearman correlation: {corr_raw:.3f}")
else:
    print("No foreground pixels found for correlation")

# %%

