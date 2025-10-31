# %% IMPORTS
import pandas as pd

# %% LOAD CSVs
HE_csv = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/HE_10_class_dataset.csv')
KI67_csv = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset.csv')

# %% CREATE TABLE WITH COMMON CASE_IDs
# Find common case_ids
common_case_ids = set(HE_csv['case_id']) & set(KI67_csv['case_id'])

# Filter both datasets to only include common case_ids
HE_common = HE_csv[HE_csv['case_id'].isin(common_case_ids)]
KI67_common = KI67_csv[KI67_csv['case_id'].isin(common_case_ids)]

# Group by label and calculate statistics
results = []
for label in sorted(HE_common['label'].unique()):
    HE_label = HE_common[HE_common['label'] == label]
    KI67_label = KI67_common[KI67_common['label'] == label]
    
    number_of_subjects = HE_label['case_id'].nunique()
    num_slide_id_HE = HE_label['slide_id'].nunique()
    num_slide_id_KI67 = KI67_label['slide_id'].nunique()
    total_slides = num_slide_id_HE + num_slide_id_KI67
    
    results.append({
        'label': label,
        'number_of_subjects': number_of_subjects,
        'num_slide_id_HE': num_slide_id_HE,
        'num_slide_id_KI67': num_slide_id_KI67,
        'total_slides': total_slides
    })

# Create DataFrame
summary_table = pd.DataFrame(results)
summary_table = summary_table.sort_values('total_slides', ascending=False).reset_index(drop=True)

# Add Total row
total_row = pd.DataFrame([{
    'label': 'Total',
    'number_of_subjects': summary_table['number_of_subjects'].sum(),
    'num_slide_id_HE': summary_table['num_slide_id_HE'].sum(),
    'num_slide_id_KI67': summary_table['num_slide_id_KI67'].sum(),
    'total_slides': summary_table['total_slides'].sum()
}])
summary_table = pd.concat([summary_table, total_row], ignore_index=True)

print(summary_table)

# %% 