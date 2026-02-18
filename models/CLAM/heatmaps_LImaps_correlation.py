# %% IMPORTS
from PIL import Image, ImageOps
import numpy as np
from scipy.stats import spearmanr

## Increase image reading size limit
Image.MAX_IMAGE_PIXELS = 1000000000

# %% LOAD PATHS
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

# %% CALCULATE CORRELATIONS ON NON-NORMALIZED PIXELS
# extract foreground pixels only 
heatmap_bg_mask = heatmap_resized_array <= 5
li_map_bg_mask = li_map_array == 0
combined_bg_mask = heatmap_bg_mask | li_map_bg_mask
heatmap_fg_raw = heatmap_resized_array[~combined_bg_mask]
li_map_fg_raw = li_map_array[~combined_bg_mask]

# calculate Spearman correlation for non-normalized pixels
if len(heatmap_fg_raw) > 0:
    corr_raw, pval_raw = spearmanr(heatmap_fg_raw, li_map_fg_raw)
    print(f"Spearman correlation (non-normalized pixels): {corr_raw:.6f} (p-value: {pval_raw:.6e})")
else:
    print("No foreground pixels found for non-normalized correlation")

# %% CALCULATE CORRELATIONS ON NORMALIZED PIXELS
# normalize heatmap foreground pixels only
foreground_vals_heatmap = heatmap_resized_array[~heatmap_bg_mask]
if len(foreground_vals_heatmap) > 0 and np.max(foreground_vals_heatmap) > 0:
    heatmap_normd = heatmap_resized_array / np.max(foreground_vals_heatmap)
    heatmap_normd[heatmap_normd > 1] = 1
else:
    heatmap_normd = heatmap_resized_array / 255.0 if np.max(heatmap_resized_array) > 0 else heatmap_resized_array

heatmap_normd[heatmap_bg_mask] = 0

# normalize LI map foreground pixel only (95th percentile)
relevant_values = li_map_array[~li_map_bg_mask]
if len(relevant_values) > 0:
    n_relval = len(relevant_values)
    adjusted_n_relval = round(n_relval * 0.95)
    sorted_relval = sorted(relevant_values)
    cm_lim = max(sorted_relval[:adjusted_n_relval]) if adjusted_n_relval > 0 else np.max(li_map_array)
    
    li_map_normd = li_map_array / cm_lim
    li_map_normd[li_map_normd > 1] = 1
else:
    li_map_normd = li_map_array / np.max(li_map_array) if np.max(li_map_array) > 0 else li_map_array

li_map_normd[li_map_bg_mask] = 0

# extract normalized foreground pixels
heatmap_fg_normd = heatmap_normd[~combined_bg_mask]
li_map_fg_normd = li_map_normd[~combined_bg_mask]

# calculate Spearman correlation for normalized pixels
if len(heatmap_fg_normd) > 0:
    corr_normd, pval_normd = spearmanr(heatmap_fg_normd, li_map_fg_normd)
    print(f"Spearman correlation (normalized pixels): {corr_normd:.6f} (p-value: {pval_normd:.6e})")
else:
    print("No foreground pixels found for normalized correlation")

# %%

