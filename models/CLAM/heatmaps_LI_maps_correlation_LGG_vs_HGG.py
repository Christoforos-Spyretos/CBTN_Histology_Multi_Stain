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
path_to_LGG_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/LGG_vs_HGG/KI67/LGG'
path_to_HGG_KI67_heatmaps = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/LGG_vs_HGG/KI67/HGG'
path_to_density_and_KI67_LI_maps = '/local/data1/chrsp39/QuPath_Portable/CBTN_Results/Normalized_Density_Maps_2'
ki67_li = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/QuPath_Ki-67_summary_analysis.csv')

# output directory for CSVs
output_dir = '/local/data3/chrsp39/CBTN_v2/CSVs'
os.makedirs(output_dir, exist_ok=True)

# %% PREPARE DATA
# add session parsed from slide_id
ki67_li['session'] = ki67_li['slide_id'].apply(lambda s: s.split('___')[1] if isinstance(s, str) and '___' in s else '')

# select and reorder columns, then rename percentage column
ki67_li = ki67_li[['case_id', 'session', 'slide_id', 'label', 'Positive', 'Pos_Density', 'Negative', 'Neg_Density', 'Pos_Percentage']]
ki67_li.rename(columns={'Pos_Percentage': 'ki67_li'}, inplace=True)

# extract slide_id from heatmap filename
def extract_slide_id(heatmap_path):
    basename = os.path.basename(heatmap_path)
    parts = basename.split('_0.5_roi_')[0]
    return parts

# parse slide_id into case_id and session (e.g. C401964___7316-1779___Ki-67)
def parse_slide_id(slide_id):
    parts = slide_id.split('___')
    case_id = parts[0] if len(parts) > 0 else ''
    session = parts[1] if len(parts) > 1 else ''
    return case_id, session

# get LI map path from slide_id
def get_li_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_Ki67_LI_map.png')

# get negative density map path from slide_id
def get_negative_density_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_NegDMap.png')

# get positive density map path from slide_id
def get_positive_density_map_path(slide_id):
    return os.path.join(path_to_density_and_KI67_LI_maps, f'{slide_id}_PosDMap.png')

# collect heatmap paths per label
heatmap_suffix = '_0.5_roi_0_blur_1_rs_0_bc_1_a_1.0_l_1_bi_0_-1.0.jpg'
label_paths = {
    'LGG': [f for f in glob(os.path.join(path_to_LGG_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
    'HGG': [f for f in glob(os.path.join(path_to_HGG_KI67_heatmaps, '*', '*.jpg')) if f.endswith(heatmap_suffix)],
}

# %% CORRELATION ANALYSIS BETWEEN HEATMAPS AND LI MAPS
print("Correlation analysis between heatmaps and KI67 LI maps:")

# Store correlations
results_LI_maps = {}
results_li_per_slide = []
all_heatmap_fg_raw = []
all_li_map_fg_raw = []

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

        # per-slide Spearman correlation
        if len(heatmap_fg_raw) > 0:
            corr_slide, p_slide = spearmanr(heatmap_fg_raw, li_map_fg_raw)
            case_id, session = parse_slide_id(slide_id)
            results_li_per_slide.append({'case_id': case_id, 'session': session, 'slide_id': slide_id, 'label': label, 'rho': float(corr_slide)})

    # calculate Spearman correlation
    if len(label_heatmap_fg_raw) > 0:
        corr_raw, pval_raw = spearmanr(label_heatmap_fg_raw, label_li_map_fg_raw)
        print(f"[{label}] Spearman correlation: {corr_raw:.3f}")
        results_LI_maps[label] = {'raw_corr': corr_raw}
    else:
        print(f"[{label}] No foreground pixels found for correlation")
        results_LI_maps[label] = {'raw_corr': None}

if len(all_heatmap_fg_raw) > 0:
    corr_raw_total, pval_raw_total = spearmanr(all_heatmap_fg_raw, all_li_map_fg_raw)
    print(f"[TOTAL] Spearman correlation pixels: {corr_raw_total:.3f}")
else:
    print("[TOTAL] No foreground pixels found for correlation")

# save per-slide LI correlations
if len(results_li_per_slide) > 0:
    df_li = pd.DataFrame(results_li_per_slide)
    # enforce column order
    df_li = df_li[['case_id', 'session', 'slide_id', 'label', 'rho']]
    df_li.to_csv(os.path.join(output_dir, 'spearman_heatmap_vs_ki67_li_map_per_slide_LGG_vs_HGG.csv'), index=False)
else:
    print("No per-slide LI correlations to save")

# correlation between LI map correlations and Ki67 LI
if len(results_li_per_slide) > 0:
    # merge per-slide rho with ki67_li values (match on case_id, session, slide_id)
    merged_li = df_li.merge(ki67_li[['case_id', 'session', 'slide_id', 'ki67_li']],
                            on=['case_id', 'session', 'slide_id'], how='inner')
    if merged_li.empty:
        print('No matching entries between per-slide LI rhos and ki67_li')

    # also produce a per-slide merge (keep only slides present in both)
    merged_by_slide = df_li.merge(ki67_li[['slide_id', 'Positive', 'Pos_Density', 'Negative', 'Neg_Density', 'ki67_li']], on='slide_id', how='inner')
    if merged_by_slide.empty:
        print('No slides matched between per-slide LI rhos and ki67_li for per-slide output')
    else:
        # keep relevant columns and rename rho
        merged_by_slide = merged_by_slide[['case_id', 'session', 'slide_id', 'label', 'rho', 'Positive', 'Pos_Density', 'Negative', 'Neg_Density', 'ki67_li']]
        merged_by_slide = merged_by_slide.rename(columns={'rho': 'rho_per_slide'})

        # helper to compute spearman + bootstrap CI
        def spearman_with_bootstrap(x, y, boot_iters=1000, rng=None):
            if rng is None:
                rng = np.random.default_rng()
            # drop pairs with NaN
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

        # compute correlations for requested Ki-67 columns
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

        # collect global rhos into a dict and add columns to the per-slide DF
        summary = pd.DataFrame(summary_rows)
        rho_map = {}
        for row in summary_rows:
            # metric is like 'spearman_rho_per_slide_vs_ki67_li' or '..._Positive'
            metric = row.get('metric')
            # extract suffix after last '_vs_'
            if metric and '_vs_' in metric:
                suffix = metric.split('_vs_')[-1]
            else:
                suffix = metric
            colname = f'rho_vs_{suffix}'
            try:
                rho_val = float(row['rho']) if row['rho'] != '' else np.nan
            except Exception:
                rho_val = np.nan
            rho_map[suffix] = rho_val
            # add a column filled with the global rho for this metric
            merged_by_slide[colname] = rho_val

        # save per-slide merged file (now includes rho_vs_* columns)
        out_path = os.path.join(output_dir, 'spearman_spearman_heatmap_and_li_map_vs_ki67_per_slide_LGG_vs_HGG.csv')
        merged_by_slide.to_csv(out_path, index=False)

        # save summary with all metrics
        summary.to_csv(os.path.join(output_dir, 'spearman_heatmap_vs_ki67_summary_LGG_vs_HGG.csv'), index=False)

        # save scatter plots for each metric
        for metric_name, col in metrics:
            try:
                x = merged_by_slide['rho_per_slide'].to_numpy()
                y = merged_by_slide[col].to_numpy()
                # require at least 2 paired points
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
                fname = os.path.join(output_dir, f'spearman_heatmap_vs_{col}_scatter_LGG_vs_HGG.png')
                plt.savefig(fname)
                plt.close()
            except Exception:
                print(f"Failed to create scatter for {col}")

        print(f"Saved per-slide merged CSV to {out_path} and summary/plots")

# %% CORRELATION ANALYSIS BETWEEN HEATMAPS AND NEGATIVE DENSITY MAPS
print("Correlation analysis between heatmaps and negative density maps:")

results_negative_density_maps = {}
results_neg_density_per_slide = []
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
        # per-slide Spearman correlation
        if len(heatmap_fg_raw) > 0:
            corr_slide, p_slide = spearmanr(heatmap_fg_raw, neg_density_map_fg_raw)
            case_id, session = parse_slide_id(slide_id)
            results_neg_density_per_slide.append({'case_id': case_id, 'session': session, 'slide_id': slide_id, 'label': label, 'rho': float(corr_slide)})

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

# save per-slide negative density correlations
if len(results_neg_density_per_slide) > 0:
    df_neg = pd.DataFrame(results_neg_density_per_slide)
    df_neg = df_neg[['case_id', 'session', 'slide_id', 'label', 'rho']]
    df_neg.to_csv(os.path.join(output_dir, 'spearman_negative_density_per_slide_LGG_vs_HGG.csv'), index=False)
    print(f"Saved per-slide negative density correlations to {os.path.join(output_dir, 'spearman_negative_density_per_slide_LGG_vs_HGG.csv')} ({len(df_neg)} rows)")
    # per-slide merge with ki67_li
    merged_neg_by_slide = df_neg.merge(ki67_li[['slide_id', 'ki67_li']], on='slide_id', how='inner')
    if not merged_neg_by_slide.empty:
        merged_neg_by_slide.to_csv(os.path.join(output_dir, 'spearman_neg_density_vs_ki67_per_slide_LGG_vs_HGG.csv'), index=False)
        try:
            gc, gp = spearmanr(merged_neg_by_slide['rho'], ki67_li.set_index('slide_id').loc[merged_neg_by_slide['slide_id'], 'ki67_li'])
            print(f"Global Spearman (per-slide) between negative-density rho and ki67_li: {gc:.3f} (n={len(merged_neg_by_slide)})")
        except Exception:
            print('Could not compute global Spearman for negative-density per-slide merge')
else:
    print("No per-slide negative density correlations to save")

# %% CORRELATION ANALYSIS BETWEEN HEATMAPS AND POSITIVE DENSITY MAPS
print("Correlation analysis between heatmaps and positive density maps:")

results_positive_density_maps = {}
results_pos_density_per_slide = []
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
        # per-slide Spearman correlation
        if len(heatmap_fg_raw) > 0:
            corr_slide, p_slide = spearmanr(heatmap_fg_raw, pos_density_map_fg_raw)
            case_id, session = parse_slide_id(slide_id)
            results_pos_density_per_slide.append({'case_id': case_id, 'session': session, 'slide_id': slide_id, 'label': label, 'rho': float(corr_slide)})

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

# save per-slide positive density correlations
if len(results_pos_density_per_slide) > 0:
    df_pos = pd.DataFrame(results_pos_density_per_slide)
    df_pos = df_pos[['case_id', 'session', 'slide_id', 'label', 'rho']]
    df_pos.to_csv(os.path.join(output_dir, 'spearman_positive_density_per_slide_LGG_vs_HGG.csv'), index=False)
    print(f"Saved per-slide positive density correlations to {os.path.join(output_dir, 'spearman_positive_density_per_slide_LGG_vs_HGG.csv')} ({len(df_pos)} rows)")
    # per-slide merge with ki67_li
    merged_pos_by_slide = df_pos.merge(ki67_li[['slide_id', 'ki67_li']], on='slide_id', how='inner')
    if not merged_pos_by_slide.empty:
        merged_pos_by_slide.to_csv(os.path.join(output_dir, 'spearman_pos_density_vs_ki67_per_slide_LGG_vs_HGG.csv'), index=False)
        try:
            gc2, gp2 = spearmanr(merged_pos_by_slide['rho'], ki67_li.set_index('slide_id').loc[merged_pos_by_slide['slide_id'], 'ki67_li'])
            print(f"Global Spearman (per-slide) between positive-density rho and ki67_li: {gc2:.3f} (n={len(merged_pos_by_slide)})")
        except Exception:
            print('Could not compute global Spearman for positive-density per-slide merge')
else:
    print("No per-slide positive density correlations to save")

# %% SINGLE IMAGE CORRELATION
path_to_KI67_heatmap = '/local/data3/chrsp39/CBTN_v2/ATTENTION_MAPS/LGG_vs_HGG/KI67/LGG/C28659___7316-84___Ki-67/C28659___7316-84___Ki-67_0.5_roi_0_blur_1_rs_0_bc_1_a_1.0_l_1_bi_0_-1.0.jpg'
path_to_KI67_LI_map = '/local/data1/chrsp39/QuPath_Portable/CBTN_Results/Normalized_Density_Maps_2/C28659___7316-84___Ki-67_Ki67_LI_map.png'
path_to_KI67_NegDMap = '/local/data1/chrsp39/QuPath_Portable/CBTN_Results/Normalized_Density_Maps_2/C28659___7316-84___Ki-67_NegDMap.png'
path_to_KI67_PosDMap = '/local/data1/chrsp39/QuPath_Portable/CBTN_Results/Normalized_Density_Maps_2/C28659___7316-84___Ki-67_PosDMap.png'

# load heatmap
heatmap_im = ImageOps.grayscale(Image.open(path_to_KI67_heatmap))
heatmap_array = np.array(heatmap_im).astype(float)

# load KI67 LI map
li_map_im = Image.open(path_to_KI67_LI_map)
if li_map_im.mode != 'L':
    li_map_im = li_map_im.convert('L')

li_width, li_height = li_map_im.size

neg_map_im = Image.open(path_to_KI67_NegDMap)
if neg_map_im.mode != 'L':
    neg_map_im = neg_map_im.convert('L')

negd_width, negd_height = neg_map_im.size

pos_map_im = Image.open(path_to_KI67_PosDMap)
if pos_map_im.mode != 'L':
    pos_map_im = pos_map_im.convert('L')

posd_width, posd_height = pos_map_im.size

# resize heatmap to match LI map dimensions
heatmap_resized = heatmap_im.resize((li_width, li_height))
heatmap_resized_array = np.array(heatmap_resized).astype(float)

# resize heatmap to match negative density map dimensions
heatmap_resized_negd = heatmap_im.resize((negd_width, negd_height))
heatmap_resized_negd_array = np.array(heatmap_resized_negd).astype(float)

# resize heatmap to match positive density map dimensions
heatmap_resized_posd = heatmap_im.resize((posd_width, posd_height))
heatmap_resized_posd_array = np.array(heatmap_resized_posd).astype(float)

# LI map array
li_map_array = np.array(li_map_im).astype(float)

negd_map_array = np.array(neg_map_im).astype(float)

posd_map_array = np.array(pos_map_im).astype(float)

# extract foreground pixels only and compute per-pair correlations
heatmap_bg_mask = heatmap_resized_array <= 5

li_map_bg_mask = li_map_array == 0
combined_bg_mask = heatmap_bg_mask | li_map_bg_mask
heatmap_fg_li = heatmap_resized_array[~combined_bg_mask]
li_map_fg_raw = li_map_array[~combined_bg_mask]

if len(heatmap_fg_li) > 0:
    corr_raw, pval_raw = spearmanr(heatmap_fg_li, li_map_fg_raw)
    print(f"Spearman correlation between heatmap and LI map: {corr_raw:.3f}")
else:
    print("No foreground pixels found for LI correlation")

negd_map_bg_mask = negd_map_array == 0
combined_bg_mask = heatmap_bg_mask | negd_map_bg_mask
heatmap_fg_negd = heatmap_resized_array[~combined_bg_mask]
negd_map_fg_raw = negd_map_array[~combined_bg_mask]

if len(heatmap_fg_negd) > 0:
    corr_raw, pval_raw = spearmanr(heatmap_fg_negd, negd_map_fg_raw)
    print(f"Spearman correlation between heatmap and negative density map: {corr_raw:.3f}")
else:
    print("No foreground pixels found for negative density correlation")

posd_map_bg_mask = posd_map_array == 0
combined_bg_mask = heatmap_bg_mask | posd_map_bg_mask
heatmap_fg_posd = heatmap_resized_array[~combined_bg_mask]
posd_map_fg_raw = posd_map_array[~combined_bg_mask]

if len(heatmap_fg_posd) > 0:
    corr_raw, pval_raw = spearmanr(heatmap_fg_posd, posd_map_fg_raw)
    print(f"Spearman correlation between heatmap and positive density map: {corr_raw:.3f}")
else:
    print("No foreground pixels found for positive density correlation")

# 2x2 figure of pixel distributions (foreground only)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
(ax11, ax12), (ax21, ax22) = axes

if len(heatmap_fg_li) > 0:
    ax11.hist(heatmap_fg_li, bins=50, alpha=0.8)
    ax11.set_title('Heatmap (LI mask)')
    ax11.set_xlabel('intensity')
    ax11.set_ylabel('count')

if len(li_map_fg_raw) > 0:
    ax12.hist(li_map_fg_raw, bins=50, alpha=0.8, color='orange')
    ax12.set_title('LI map')
    ax12.set_xlabel('intensity')
    ax12.set_ylabel('count')

if len(negd_map_fg_raw) > 0:
    ax21.hist(negd_map_fg_raw, bins=50, alpha=0.8, color='green')
    ax21.set_title('Negative density map')
    ax21.set_xlabel('intensity')
    ax21.set_ylabel('count')

if len(posd_map_fg_raw) > 0:
    ax22.hist(posd_map_fg_raw, bins=50, alpha=0.8, color='red')
    ax22.set_title('Positive density map')
    ax22.set_xlabel('intensity')
    ax22.set_ylabel('count')

fig.suptitle('Foreground pixel distributions (single_LGG_vs_HGG)')
fig.tight_layout(rect=[0, 0, 1, 0.95])
out_path = os.path.join(output_dir, 'single_LGG_vs_HGG_pixel_distributions_2x2.png')
fig.savefig(out_path)
plt.close(fig)

# %%

