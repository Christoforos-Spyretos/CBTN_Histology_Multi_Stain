# %% IMPORTS
from PIL import Image, ImageOps
import numpy as np
from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt
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

# load ki67 summary and set output dir
ki67_li = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/QuPath_Ki-67_summary_analysis.csv')
output_dir = '/local/data3/chrsp39/CBTN_v2/CSVs'
os.makedirs(output_dir, exist_ok=True)

# add session parsed from slide_id and keep/rename columns
ki67_li['session'] = ki67_li['slide_id'].apply(lambda s: s.split('___')[1] if isinstance(s, str) and '___' in s else '')
ki67_li = ki67_li[['case_id', 'session', 'slide_id', 'label', 'Positive', 'Pos_Density', 'Negative', 'Neg_Density', 'Pos_Percentage']]
ki67_li.rename(columns={'Pos_Percentage': 'ki67_li'}, inplace=True)

# extract slide_id from heatmap filename
def extract_slide_id(heatmap_path):
    basename = os.path.basename(heatmap_path)
    parts = basename.split('_0.5_roi_')[0]
    return parts

# get LI map path from slide_id
def get_li_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_Ki67_LI_map.png')

# get negative density map path from slide_id
def get_negative_density_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_NegDMap.png')

# get positive density map path from slide_id
def get_positive_density_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_PosDMap.png')

# collect heatmap paths per label (filter by suffix)
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

# per-slide results
results_li_per_slide = []

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
        # save per-slide LI correlations
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

        # per-slide Spearman correlation
        if len(heatmap_fg_raw) > 0:
            corr_slide, p_slide = spearmanr(heatmap_fg_raw, li_map_fg_raw)
            # parse case/session and append
            # slide_id format expected like 'C401964___7316-1779___Ki-67'
            parts = slide_id.split('___')
            case_id = parts[0] if len(parts) > 0 else ''
            session = parts[1] if len(parts) > 1 else ''
            results_li_per_slide.append({'case_id': case_id, 'session': session, 'slide_id': slide_id, 'label': label, 'rho': float(corr_slide)})

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

# save per-slide LI correlations and correlate with Ki-67 metrics
if len(results_li_per_slide) > 0:
    df_li = pd.DataFrame(results_li_per_slide)
    df_li = df_li[['case_id', 'session', 'slide_id', 'label', 'rho']]
    df_li.to_csv(os.path.join(output_dir, 'spearman_heatmap_vs_ki67_li_map_per_slide_5_class.csv'), index=False)
    print(f"Saved per-slide LI correlations to {os.path.join(output_dir, 'spearman_heatmap_vs_ki67_li_map_per_slide_5_class.csv')} ({len(df_li)} rows)")

    merged_by_slide = df_li.merge(ki67_li[['slide_id', 'Positive', 'Pos_Density', 'Negative', 'Neg_Density', 'ki67_li']], on='slide_id', how='inner')
    if not merged_by_slide.empty:
        merged_by_slide = merged_by_slide[['case_id', 'session', 'slide_id', 'label', 'rho', 'Positive', 'Pos_Density', 'Negative', 'Neg_Density', 'ki67_li']]
        merged_by_slide = merged_by_slide.rename(columns={'rho': 'rho_per_slide'})

        def spearman_with_bootstrap(x, y, boot_iters=1000, rng=None):
            if rng is None:
                rng = np.random.default_rng()
            mask = (~np.isnan(x)) & (~np.isnan(y))
            x = x[mask]
            y = y[mask]
            n = len(x)
            if n < 2:
                return (np.nan, np.nan, np.nan, np.nan, n)
            try:
                rho, pval = spearmanr(x, y)
            except Exception:
                rho, pval = (np.nan, np.nan)
            boots = []
            for _ in range(boot_iters):
                idx = rng.integers(0, n, n)
                try:
                    bc, _ = spearmanr(x[idx], y[idx])
                    if not np.isnan(bc):
                        boots.append(bc)
                except Exception:
                    continue
            if len(boots) > 0:
                ci_lower, ci_upper = np.nanpercentile(boots, [2.5, 97.5])
            else:
                ci_lower, ci_upper = (np.nan, np.nan)
            return (rho, pval, ci_lower, ci_upper, n)

        rng = np.random.default_rng()
        metrics = [
            ('ki67_li', 'ki67_li'),
            ('Positive', 'Positive'),
            ('Pos_Density', 'Pos_Density'),
            ('Negative', 'Negative'),
            ('Neg_Density', 'Neg_Density'),
        ]
        summary_rows = []
        for metric_name, col in metrics:
            x = merged_by_slide['rho_per_slide'].to_numpy(dtype=float)
            y = merged_by_slide[col].to_numpy(dtype=float)
            rho, pval, ci_lower, ci_upper, n_matched = spearman_with_bootstrap(x, y, boot_iters=1000, rng=rng)
            if not np.isnan(rho):
                print(f"Global Spearman (per-slide) between LI rho and {col}: {rho:.3f} (n={n_matched})")
            else:
                print(f"Could not compute Spearman between LI rho and {col} (n={n_matched})")
            summary_rows.append({'metric': f'spearman_rho_per_slide_vs_{metric_name}', 'rho': float(rho) if not np.isnan(rho) else '', 'pval': float(pval) if not np.isnan(pval) else '', 'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else '', 'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else '', 'n': int(n_matched)})

        summary = pd.DataFrame(summary_rows)
        rho_map = {}
        for row in summary_rows:
            metric = row.get('metric')
            if metric and '_vs_' in metric:
                suffix = metric.split('_vs_')[-1]
            else:
                suffix = metric
            try:
                rho_val = float(row['rho']) if row['rho'] != '' else np.nan
            except Exception:
                rho_val = np.nan
            rho_map[suffix] = rho_val
            merged_by_slide[f'rho_vs_{suffix}'] = rho_val

        out_path = os.path.join(output_dir, 'spearman_spearman_heatmap_and_li_map_vs_ki67_per_slide_5_class.csv')
        merged_by_slide.to_csv(out_path, index=False)
        summary.to_csv(os.path.join(output_dir, 'spearman_heatmap_vs_ki67_summary_5_class.csv'), index=False)

        for metric_name, col in metrics:
            try:
                x = merged_by_slide['rho_per_slide'].to_numpy()
                y = merged_by_slide[col].to_numpy()
                mask = (~np.isnan(x)) & (~np.isnan(y))
                if mask.sum() < 2:
                    print(f"Not enough points to plot scatter for {col} (n={mask.sum()})")
                    continue
                rho_val = rho_map.get(col if col != 'ki67_li' else 'ki67_li', np.nan)
                title = f'{col} vs rho_per_slide (Spearman={rho_val:.3f})' if not np.isnan(rho_val) else f'{col} vs rho_per_slide'
                plt.figure(figsize=(6,4))
                plt.scatter(x[mask], y[mask], alpha=0.7)
                plt.xlabel('rho_per_slide (heatmap vs LI)')
                plt.ylabel(col)
                plt.title(f"{title}, n={int(mask.sum())}")
                plt.tight_layout()
                fname = os.path.join(output_dir, f'spearman_heatmap_vs_{col}_scatter_5_class.png')
                plt.savefig(fname)
                plt.close()
                print(f"Saved scatter for {col} to {fname}")
            except Exception:
                print(f"Failed to create scatter for {col}")
        print(f"Saved per-slide merged CSV to {out_path} and summary/plots")
    else:
        print('No slides matched between per-slide LI rhos and ki67_li for per-slide output')

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