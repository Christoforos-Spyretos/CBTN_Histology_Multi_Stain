# %% IMPORTS
import pandas as pd

# %% LOAD CSV & PATHS
ki67_li = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/QuPath_Ki-67_summary_analysis.csv')
HE_5_class = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/HE_5_class_dataset.csv')
output_csv = '/local/data3/chrsp39/CBTN_v2/CSVs/HE_5_class_aligned_KI67_LI.csv'

# %% CROSS ALIGNED CSVs
ki67_li = ki67_li[['case_id', 'slide_id', 'label', 'Pos_Percentage', 'age_at_diagnosis_(days)', 'sex']]

def extract_session(slide_id: str):
	"""Extract session token from slide_id formatted as CASE___SESSION___STAIN."""
	if pd.isna(slide_id):
		return pd.NA
	parts = str(slide_id).split('___')
	return parts[1] if len(parts) > 1 else pd.NA


ki67_li['session'] = ki67_li['slide_id'].apply(extract_session)
HE_5_class['session'] = HE_5_class['slide_id'].apply(extract_session)

ki67_for_merge = (
	ki67_li[['case_id', 'session', 'Pos_Percentage', 'age_at_diagnosis_(days)', 'sex']]
	.rename(
		columns={
			'Pos_Percentage': 'pos_percentage',
		}
	)
	.drop_duplicates(subset=['case_id', 'session'])
)

aligned_df = HE_5_class.merge(
	ki67_for_merge,
	on=['case_id', 'session'],
	how='left',
)

aligned_df = aligned_df.dropna(subset=['pos_percentage'])
aligned_df = aligned_df.drop(columns=['session'])

if 'case_id' in aligned_df.columns and 'slide_id' in aligned_df.columns:
	ordered_cols = aligned_df.columns.tolist()
	ordered_cols.remove('slide_id')
	case_id_idx = ordered_cols.index('case_id')
	ordered_cols.insert(case_id_idx + 1, 'slide_id')
	aligned_df = aligned_df[ordered_cols]

# Print unique counts for aligned_df
unique_case = aligned_df['case_id'].nunique(dropna=True)
unique_slide = aligned_df['slide_id'].nunique(dropna=True)
print(f"unique_case_id: {unique_case}")
print(f"unique_slide_id: {unique_slide}")

if 'label' in aligned_df.columns:
	label_counts = aligned_df['label'].value_counts(dropna=False)
	print("label_counts:")
	for lbl, cnt in label_counts.items():
		print(f"{lbl}: {cnt}")

aligned_df.to_csv(output_csv, index=False)

# %%