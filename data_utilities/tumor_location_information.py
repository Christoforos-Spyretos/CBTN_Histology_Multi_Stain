# %% IMPORTS
import pandas as pd
import os
import matplotlib.pyplot as plt 

# %% LOAD CSVs
df = pd.read_csv('/local/data3/chrsp39/CBTN_v2/CSVs/Merged_HE_KI67_5_class_dataset.csv')
histological_diagnosis_df_from_portal = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Histological Diagnoses', engine='openpyxl')
participants_df_from_portal = pd.read_excel("/local/data3/chrsp39/CBTN_v2/CSVs/CBTN_clinical_data_from_portal.xlsx", sheet_name='Participants', engine='openpyxl')

# %% MATCH CASE_ID AND EXTRACT TUMOR LOCATION
# Map the age at diagnosis to df based on case_id
df['age_at_diagnosis_(days)'] = df['case_id'].map(histological_diagnosis_df_from_portal.set_index('External Id')['Age at Diagnosis (Days)'].to_dict())

df['gender'] = df['case_id'].map(participants_df_from_portal.set_index('External Id')['Gender'].to_dict())

# Convert age from days to years
df['age_at_diagnosis_(years)'] = df['age_at_diagnosis_(days)'] / 365.25

# Map the tumor location to df based on case_id
df['tumor_location'] = df['case_id'].map(histological_diagnosis_df_from_portal.set_index('External Id')['Histological Tumor Location'].to_dict())

# %% PRINT UNIQUE TUMOR LOCATIONS WITH COUNTS
print("Unique Tumor Locations and their counts:")
print("="*50)
tumor_location_counts = df['tumor_location'].value_counts()
print(tumor_location_counts)
print(f"\nTotal unique tumor locations: {len(tumor_location_counts)}")

# %% PRINT UNIQUE COMBINATIONS OF LABEL AND TUMOR LOCATION
print("\n\nUnique combinations of Label and Tumor Location:")
print("="*50)
label_location_combinations = df.groupby(['label', 'tumor_location']).size().reset_index(name='count')
label_location_combinations = label_location_combinations.sort_values(['label', 'count'], ascending=[True, False])
print(label_location_combinations.to_string(index=False))
print(f"\nTotal unique combinations: {len(label_location_combinations)}")

# %%
