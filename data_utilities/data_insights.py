# %% IMPORTS
import pandas as pd
import os

# %% LOAD CSVs
HE_csv = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/HE_10_class_dataset.csv')
KI67_csv = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/KI67_10_class_dataset.csv')
histological_diagnosis_df_from_portal = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Histological Diagnoses', engine='openpyxl')
participants_df_from_portal = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Participants', engine='openpyxl')

# %% LABEL SUMMARY
common_case_ids = set(HE_csv['case_id']) & set(KI67_csv['case_id'])

HE_common = HE_csv[HE_csv['case_id'].isin(common_case_ids)]
KI67_common = KI67_csv[KI67_csv['case_id'].isin(common_case_ids)]

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

summary_table = pd.DataFrame(results)
summary_table = summary_table.sort_values('total_slides', ascending=False).reset_index(drop=True)

total_row = pd.DataFrame([{
    'label': 'Total',
    'number_of_subjects': summary_table['number_of_subjects'].sum(),
    'num_slide_id_HE': summary_table['num_slide_id_HE'].sum(),
    'num_slide_id_KI67': summary_table['num_slide_id_KI67'].sum(),
    'total_slides': summary_table['total_slides'].sum()
}])
summary_table = pd.concat([summary_table, total_row], ignore_index=True)

print(summary_table)

# %% TUMOR LOCATION SUMMARY
merged_common = pd.concat([HE_common[['case_id', 'label']], KI67_common[['case_id', 'label']]]).drop_duplicates(subset='case_id')

merged_common['tumor_location'] = merged_common['case_id'].map(histological_diagnosis_df_from_portal.set_index('External Id')['Histological Tumor Location'].to_dict())

# unique tumor locations
print("\n\nUnique Tumor Locations and their counts:")
print("="*50)
tumor_location_counts = merged_common['tumor_location'].value_counts()
print(tumor_location_counts)
print(f"\nTotal unique tumor locations: {len(tumor_location_counts)}")

# unique combinations of label and tumor location
print("\n\nUnique combinations of Label and Tumor Location:")
print("="*50)
label_location_combinations = merged_common.groupby(['label', 'tumor_location']).size().reset_index(name='count')
label_location_combinations = label_location_combinations.sort_values(['label', 'count'], ascending=[True, False])
print(label_location_combinations.to_string(index=False))
print(f"\nTotal unique combinations: {len(label_location_combinations)}")

# %% GENDER DISTRIBUTION
print("\n\nGender Distribution:")
print("="*50)

merged_common['gender'] = merged_common['case_id'].map(participants_df_from_portal.set_index('External Id')['Gender'].to_dict())

gender_counts = merged_common['gender'].value_counts()
print(gender_counts)
print(f"\nTotal unique genders: {len(gender_counts)}")

# %% AGE DISTRIBUTION
print("\n\nAge Distribution (years):")
print("="*50)

merged_common['age_at_diagnosis_(days)'] = merged_common['case_id'].map(histological_diagnosis_df_from_portal.set_index('External Id')['Age at Diagnosis (Days)'].to_dict())

# convert age from days to years
merged_common['age_at_diagnosis_(years)'] = merged_common['age_at_diagnosis_(days)'] / 365.25

# total age distribution
print("\nTotal Age Distribution:")
print("-"*50)
print(f"  Mean:   {merged_common['age_at_diagnosis_(years)'].mean():.2f}")
print(f"  Std:    {merged_common['age_at_diagnosis_(years)'].std():.2f}")
print(f"  Median: {merged_common['age_at_diagnosis_(years)'].median():.2f}")
print(f"  Min:    {merged_common['age_at_diagnosis_(years)'].min():.2f}")
print(f"  Max:    {merged_common['age_at_diagnosis_(years)'].max():.2f}")

# age distribution by gender
print("\n\nAge Distribution by Gender:")
print("-"*50)
for gender in sorted(merged_common['gender'].dropna().unique()):
    gender_data = merged_common[merged_common['gender'] == gender]['age_at_diagnosis_(years)']
    print(f"\n{gender}:")
    print(f"  Count:  {len(gender_data)}")
    print(f"  Mean:   {gender_data.mean():.2f}")
    print(f"  Std:    {gender_data.std():.2f}")
    print(f"  Median: {gender_data.median():.2f}")
    print(f"  Min:    {gender_data.min():.2f}")
    print(f"  Max:    {gender_data.max():.2f}")

# age distribution by label
print("\n\nAge Distribution by Label:")
print("-"*50)
for label in sorted(merged_common['label'].unique()):
    label_data = merged_common[merged_common['label'] == label]['age_at_diagnosis_(years)']
    print(f"\n{label}:")
    print(f"  Count:  {len(label_data)}")
    print(f"  Mean:   {label_data.mean():.2f}")
    print(f"  Std:    {label_data.std():.2f}")
    print(f"  Median: {label_data.median():.2f}")
    print(f"  Min:    {label_data.min():.2f}")
    print(f"  Max:    {label_data.max():.2f}")

# %%

