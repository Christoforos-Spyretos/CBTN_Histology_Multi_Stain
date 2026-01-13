# %% IMPORTS
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd

## Increase image reading size limit
Image.MAX_IMAGE_PIXELS = 1000000000

# %% UTILITY FUNCTIONS
def get_histogram(img):
    '''
    Get the histogram of an image. For an 8-bit, grayscale image, the
    histogram will be a 256 unit vector in which the nth value indicates
    the percent of the pixels in the image with the given darkness level.
    The histogram's values sum to 1.
    '''
    h, w = img.shape
    hist = [0.0] * 256
    bg = 0
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
                
    return np.array(hist) / (h * w) 

def normalize_exposure(img):
    '''
    Normalize the exposure of an image.
    '''
    img = img*255
    img = img.astype(int)
    hist = get_histogram(img)
    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)

def earth_movers_distance(img_a, img_b):
    '''
    Measure the Earth Mover's distance between two images
    @args:
        {str} img_a: image array a
        {str} img_b: image array b
    @returns:
        wasserstein_distance between the two images
    '''
    img_a = normalize_exposure(img_a)
    img_b = normalize_exposure(img_b)
    hist_a = get_histogram(img_a)
    hist_b = get_histogram(img_b)
    return wasserstein_distance(hist_a, hist_b)

# %% LOAD PATHS AND FILES
## Define working directories and files:
work_dir = os.getcwd()
heatmaps_dir = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/LGG_vs_HGG/KI67/HEATMAPS'
density_maps_dir = '/local/data1/chrsp39/QuPath_Portable/CBTN_Results/Normalized_Density_Maps_2'
csv = '/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv'
save_dir = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/LGG_vs_HGG/KI67/COMPARISON_MAPS'

if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# %% MAIN SCRIPT
label_df = pd.read_csv(csv)
label_dict = dict(zip(label_df['slide_id'], label_df['label']))

# Create result dataframe with columns for all three density map comparisons:
res_df = pd.DataFrame(columns = ['slide', 'label', 
                                  'MSE_LI', 'SSIM_LI', 'EMD_LI',
                                  'MSE_Neg', 'SSIM_Neg', 'EMD_Neg',
                                  'MSE_Pos', 'SSIM_Pos', 'EMD_Pos'])

# Get all heatmap files
heatmap_files = [f for f in os.listdir(heatmaps_dir) if f.endswith('.jpg') or f.endswith('.tif')]

for heatmap_file in heatmap_files:
    
    # Extract base name from heatmap (everything before _0.5_roi_...)
    base_name = heatmap_file.split('_0.5_roi_')[0]
    
    # Extract slide_id (first part before ___)
    slide_id = base_name.split('___')[0]
    
    # Get label from the CSV
    label = label_dict.get(slide_id, 'Unknown')
    
    print(f'Processing: {base_name} (Slide ID: {slide_id}, Label: {label})')
    
    # Construct density map file names
    li_map_file = base_name + '_Ki67_LI_map.png'
    neg_map_file = base_name + '_NegDMap.png'
    pos_map_file = base_name + '_PosDMap.png'
    
    # Full paths
    heatmap_path = os.path.join(heatmaps_dir, heatmap_file)
    li_map_path = os.path.join(density_maps_dir, li_map_file)
    neg_map_path = os.path.join(density_maps_dir, neg_map_file)
    pos_map_path = os.path.join(density_maps_dir, pos_map_file)
    
    # Check if all required files exist
    if not os.path.isfile(heatmap_path):
        print(f'  Heatmap not found: {heatmap_path}')
        continue
    if not os.path.isfile(li_map_path):
        print(f'  LI map not found: {li_map_path}')
        continue
    if not os.path.isfile(neg_map_path):
        print(f'  Neg map not found: {neg_map_path}')
        continue
    if not os.path.isfile(pos_map_path):
        print(f'  Pos map not found: {pos_map_path}')
        continue
    
    ## Create current directory for this slide:
    slide_save_dir = os.path.join(save_dir, base_name)
    if not os.path.isdir(slide_save_dir):
        os.makedirs(slide_save_dir)
    
    ## Load and process heatmap (attention map):
    heatmap_im = ImageOps.grayscale(Image.open(heatmap_path))
    heatmap_array = np.array(heatmap_im)
    heatmap_normd = heatmap_array / np.max(heatmap_array) if np.max(heatmap_array) > 0 else heatmap_array
    
    heatmap_height, heatmap_width = heatmap_array.shape
    
    # Dictionary to store results for all three comparisons
    comparison_results = {}
    
    # Process each density map type
    density_maps = {
        'LI': li_map_path,
        'Neg': neg_map_path,
        'Pos': pos_map_path
    }
    
    for map_type, density_map_path in density_maps.items():
        print(f'  Comparing with {map_type} map...')
        
        ## Load density map
        density_im = Image.open(density_map_path)
                
        # Convert to grayscale if needed
        if density_im.mode != 'L':
            # If it's an RGB image, it might already be a visualization
            # Extract just one channel or convert properly
            density_im = density_im.convert('L')
        
        density_width, density_height = density_im.size
        
        ## Set color bar orientation depending on image dimensions
        if density_width < density_height:
            orient = 'vertical'
        else:
            orient = 'horizontal'
        
        ## Resize heatmap to match density map dimensions
        heatmap_resized = heatmap_im.resize((density_width, density_height))
        heatmap_resized_array = np.array(heatmap_resized).astype(float)
        
        # Create background mask based on edge pixels (background typically at edges)
        edge_value = np.median([heatmap_resized_array[0,:].mean(), heatmap_resized_array[-1,:].mean(),
                               heatmap_resized_array[:,0].mean(), heatmap_resized_array[:,-1].mean()])
        heatmap_bg_mask = np.abs(heatmap_resized_array - edge_value) < 10
        
        # Normalize using only foreground pixels
        foreground_vals = heatmap_resized_array[~heatmap_bg_mask]
        if len(foreground_vals) > 0 and np.max(foreground_vals) > 0:
            heatmap_resized_normd = heatmap_resized_array / np.max(foreground_vals)
            heatmap_resized_normd[heatmap_resized_normd > 1] = 1
        else:
            heatmap_resized_normd = heatmap_resized_array / 255.0 if np.max(heatmap_resized_array) > 0 else heatmap_resized_array
        
        # Set background to 0 (will appear blue in jet colormap)
        heatmap_resized_normd[heatmap_bg_mask] = 0
        
        ## Process density map array
        density_array = np.array(density_im).astype(float)
        
        # Create background mask based on edge pixels (background typically at edges)
        edge_value = np.median([density_array[0,:].mean(), density_array[-1,:].mean(),
                               density_array[:,0].mean(), density_array[:,-1].mean()])
        background_mask = np.abs(density_array - edge_value) < 10
        
        # Normalize density map using only foreground pixels
        relevant_values = density_array[~background_mask]
        if len(relevant_values) > 0:
            n_relval = len(relevant_values)
            adjusted_n_relval = round(n_relval * 0.95)
            sorted_relval = sorted(relevant_values)
            cm_lim = max(sorted_relval[:adjusted_n_relval]) if adjusted_n_relval > 0 else np.max(density_array)
            
            density_normd = density_array / cm_lim
            density_normd[density_normd > 1] = 1
        else:
            density_normd = density_array / np.max(density_array) if np.max(density_array) > 0 else density_array
            cm_lim = np.max(density_array)
        
        # Set background to 0 (will appear blue in jet colormap)
        density_normd[background_mask] = 0
        
        ### Comparison metrics:
        ## Flatten arrays and exclude background to compute the MSE:
        flat_density = density_normd.flatten()
        ind = np.argwhere(flat_density > 0)
        density_fg = flat_density[ind]
        flat_heatmap = heatmap_resized_normd.flatten()
        heatmap_fg = flat_heatmap[ind]
        
        mse = mean_squared_error(density_fg, heatmap_fg)
        
        ## Compute SSIM, EMD and difference map:
        str_sim = ssim(density_normd, heatmap_resized_normd, data_range=1)
        idiff = np.absolute(density_normd - heatmap_resized_normd)
        diff = idiff / np.max(idiff) if np.max(idiff) > 0 else idiff
        emd = earth_movers_distance(heatmap_resized_normd, density_normd)
        
        print(f'    MSE: {mse:.6f}, SSIM: {str_sim:.6f}, EMD: {emd:.6f}')
        
        # Store results
        comparison_results[f'MSE_{map_type}'] = mse
        comparison_results[f'SSIM_{map_type}'] = str_sim
        comparison_results[f'EMD_{map_type}'] = emd
        
        ## Save difference map:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a masked array where background pixels are masked
        combined_bg_mask = (density_normd == 0) & (heatmap_resized_normd == 0)
        diff_masked = np.ma.masked_where(combined_bg_mask, diff)
        
        # Create colormap with white for masked values
        cmap = plt.cm.jet.copy()
        cmap.set_bad('white')
        
        im = ax.imshow(diff_masked, cmap=cmap, vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax, orientation=orient)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.title(f'Difference Map (Heatmap vs {map_type})')
        plt.savefig(os.path.join(slide_save_dir, f'{base_name}_{map_type}_DiffMap.png'), 
                   bbox_inches='tight', pad_inches=0.1, facecolor='white')
        cb.remove()
        plt.close("all")
        
        del fig, ax, cb
    
    # Append results to dataframe
    new_row = pd.DataFrame([{
        'slide': base_name,
        'label': label,
        **comparison_results
    }])
    res_df = pd.concat([res_df, new_row], ignore_index=True)
    
    print(f'Completed: {base_name}\n')

## Save result csv
res_df.to_csv(os.path.join(save_dir, 'HM_comp_results.csv'), index=False)
print(f'\nResults saved to: {os.path.join(save_dir, "HM_comp_results.csv")}')

# %% MAIN SCRIPT 2 - Hybrid Method (Simple background detection + sophisticated normalization)
label_df = pd.read_csv(csv)
label_dict = dict(zip(label_df['slide_id'], label_df['label']))

# Create result dataframe with columns for all three density map comparisons:
res_df2 = pd.DataFrame(columns = ['slide', 'label', 
                                  'MSE_LI', 'SSIM_LI', 'EMD_LI',
                                  'MSE_Neg', 'SSIM_Neg', 'EMD_Neg',
                                  'MSE_Pos', 'SSIM_Pos', 'EMD_Pos'])

# Get all heatmap files
heatmap_files = [f for f in os.listdir(heatmaps_dir) if f.endswith('.jpg') or f.endswith('.tif')]

for heatmap_file in heatmap_files:
    
    # Extract base name from heatmap (everything before _0.5_roi_...)
    base_name = heatmap_file.split('_0.5_roi_')[0]
    
    # Extract slide_id (first part before ___)
    slide_id = base_name.split('___')[0]
    
    # Get label from the CSV
    label = label_dict.get(slide_id, 'Unknown')
    
    print(f'Processing (Hybrid): {base_name} (Slide ID: {slide_id}, Label: {label})')
    
    # Construct density map file names
    li_map_file = base_name + '_Ki67_LI_map.png'
    neg_map_file = base_name + '_NegDMap.png'
    pos_map_file = base_name + '_PosDMap.png'
    
    # Full paths
    heatmap_path = os.path.join(heatmaps_dir, heatmap_file)
    li_map_path = os.path.join(density_maps_dir, li_map_file)
    neg_map_path = os.path.join(density_maps_dir, neg_map_file)
    pos_map_path = os.path.join(density_maps_dir, pos_map_file)
    
    
    ## Create current directory for this slide:
    slide_save_dir2 = os.path.join(save_dir + '_hybrid', base_name)
    if not os.path.isdir(slide_save_dir2):
        os.makedirs(slide_save_dir2)
    
    ## Load and process heatmap (attention map):
    heatmap_im = ImageOps.grayscale(Image.open(heatmap_path))
    heatmap_array = np.array(heatmap_im)
    heatmap_normd = heatmap_array / np.max(heatmap_array) if np.max(heatmap_array) > 0 else heatmap_array
    
    heatmap_height, heatmap_width = heatmap_array.shape
    
    # Dictionary to store results for all three comparisons
    comparison_results = {}
    
    # Process each density map type
    density_maps = {
        'LI': li_map_path,
        'Neg': neg_map_path,
        'Pos': pos_map_path
    }
    
    for map_type, density_map_path in density_maps.items():
        print(f'  Comparing with {map_type} map...')
        
        ## Load density map
        density_im = Image.open(density_map_path)
                
        # Convert to grayscale if needed
        if density_im.mode != 'L':
            density_im = density_im.convert('L')
        
        density_width, density_height = density_im.size
        
        ## Set color bar orientation depending on image dimensions
        if density_width < density_height:
            orient = 'vertical'
        else:
            orient = 'horizontal'
        
        ## Resize heatmap to match density map dimensions
        heatmap_resized = heatmap_im.resize((density_width, density_height))
        heatmap_resized_array = np.array(heatmap_resized).astype(float)
        
        # Simple background detection: pixels <= 5 (small threshold for near-zero values)
        heatmap_bg_mask = heatmap_resized_array <= 5
        
        # Normalize using only foreground pixels (> 0)
        foreground_vals = heatmap_resized_array[~heatmap_bg_mask]
        if len(foreground_vals) > 0 and np.max(foreground_vals) > 0:
            heatmap_resized_normd = heatmap_resized_array / np.max(foreground_vals)
            heatmap_resized_normd[heatmap_resized_normd > 1] = 1
        else:
            heatmap_resized_normd = heatmap_resized_array / 255.0 if np.max(heatmap_resized_array) > 0 else heatmap_resized_array
        
        # Set background to 0
        heatmap_resized_normd[heatmap_bg_mask] = 0
        
        ## Process density map array
        density_array = np.array(density_im).astype(float)
        
        # Simple background detection: pixels == 0
        background_mask = density_array == 0
        
        # Normalize density map using only foreground pixels (> 0)
        relevant_values = density_array[~background_mask]
        if len(relevant_values) > 0:
            n_relval = len(relevant_values)
            adjusted_n_relval = round(n_relval * 0.95)
            sorted_relval = sorted(relevant_values)
            cm_lim = max(sorted_relval[:adjusted_n_relval]) if adjusted_n_relval > 0 else np.max(density_array)
            
            density_normd = density_array / cm_lim
            density_normd[density_normd > 1] = 1
        else:
            density_normd = density_array / np.max(density_array) if np.max(density_array) > 0 else density_array
            cm_lim = np.max(density_array)
        
        # Set background to 0
        density_normd[background_mask] = 0
        
        ### Comparison metrics:
        ## Flatten arrays and exclude background to compute the MSE:
        flat_density = density_normd.flatten()
        ind = np.argwhere(flat_density > 0)
        density_fg = flat_density[ind]
        flat_heatmap = heatmap_resized_normd.flatten()
        heatmap_fg = flat_heatmap[ind]
        
        mse = mean_squared_error(density_fg, heatmap_fg)
        
        ## Compute SSIM, EMD and difference map:
        str_sim = ssim(density_normd, heatmap_resized_normd, data_range=1)
        idiff = np.absolute(density_normd - heatmap_resized_normd)
        
        # Create background mask first
        combined_bg_mask = (density_normd == 0) & (heatmap_resized_normd == 0)
        
        # Normalize difference using only foreground pixels
        foreground_diff = idiff[~combined_bg_mask]
        if len(foreground_diff) > 0 and np.max(foreground_diff) > 0:
            diff = idiff / np.max(foreground_diff)
            diff[diff > 1] = 1
        else:
            diff = idiff
        
        emd = earth_movers_distance(heatmap_resized_normd, density_normd)
        
        print(f'    MSE: {mse:.6f}, SSIM: {str_sim:.6f}, EMD: {emd:.6f}')
        
        # Store results
        comparison_results[f'MSE_{map_type}'] = mse
        comparison_results[f'SSIM_{map_type}'] = str_sim
        comparison_results[f'EMD_{map_type}'] = emd
        
        ## Save difference map:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        ax.set_facecolor('white')
        
        # Use the original background masks (before normalization) for white display
        display_bg_mask = heatmap_bg_mask | background_mask
        diff_masked = np.ma.masked_where(display_bg_mask, diff)
        
        # Create colormap with white for masked values
        cmap = plt.cm.jet.copy()
        cmap.set_bad('white')
        
        im = ax.imshow(diff_masked, cmap=cmap, vmin=0, vmax=1, interpolation='none')
        cb = fig.colorbar(im, ax=ax, orientation=orient)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.title(f'Difference Map (Heatmap vs {map_type}) - Hybrid Method')
        plt.savefig(os.path.join(slide_save_dir2, f'{base_name}_{map_type}_DiffMap.png'), 
                   bbox_inches='tight', pad_inches=0.1, facecolor='white')
        cb.remove()
        plt.close("all")
        
        del fig, ax, cb
    
    # Append results to dataframe
    new_row = pd.DataFrame([{
        'slide': base_name,
        'label': label,
        **comparison_results
    }])
    res_df2 = pd.concat([res_df2, new_row], ignore_index=True)
    
    print(f'Completed: {base_name}\n')

## Save result csv
res_df2.to_csv(os.path.join(save_dir + '_hybrid', 'HM_comp_results_hybrid.csv'), index=False)
print(f'\nHybrid results saved to: {os.path.join(save_dir + "_hybrid", "HM_comp_results_hybrid.csv")}')

# %% 
