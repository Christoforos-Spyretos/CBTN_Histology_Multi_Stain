# %% IMPORTS
import csv
import os
import fnmatch
import pandas as pd

# %% DATA
# path to histology images
histology_path = "/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/HISTOLOGY/SUBJECTS"

# %% CSV HISTOLOGY FOLDER CONTENTS  
subjects_info = {}
subject_IDs = os.listdir(histology_path)

for subject_ID in subject_IDs:
    subjects_info[subject_ID] = {}
    sessions_path = os.path.join(histology_path, subject_ID, "SESSIONS")

    if os.path.exists(sessions_path):
        sessions = os.listdir(sessions_path)

        for session in sessions:
            session_path = os.path.join(sessions_path, session, "ACQUISITIONS", "Files", "FILES")

            if os.path.exists(session_path):
                svs_files = [file for file in os.listdir(session_path) if file.endswith(".svs")]

                if svs_files:
                    if subject_ID not in subjects_info:
                        subjects_info[subject_ID] = {}
                    if session not in subjects_info[subject_ID]:
                        subjects_info[subject_ID][session] = {"svs_files": []}
                    subjects_info[subject_ID][session]["svs_files"].extend(svs_files)

csv_file_path = "/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/SUMMARY_FILES/HISTOLOGY/quality_check.csv"

with open(csv_file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["subject_ID", "session", "svs"])
    for subject_ID, sessions in subjects_info.items():
        for session, session_data in sessions.items():
            svs_files = session_data.get("svs_files", [])
            if svs_files:
                for svs_file in svs_files:
                    csv_writer.writerow([subject_ID, session, svs_file])

print("CSV file exported:", csv_file_path)

# %% STAIN TYPES CSV 
# dictionary of stains
stains = {
    "H&E": ["HandE", "HandE_B1", "HandE_A1", "H_and_E_A1", "H_and_E_B1", "HandE_C1", 
            "HandE_B2", "HandE_A", "HandE_B", "HandE_A2", "H_and_E_A2", "HandE_B3",
            "HandE_C2", "HandE_C", "HandE_D1", "H_and_E_C1", "HandE_A3", "H_and_E_B2",
            "HandE_", "HandE_FSA", "HandE_FS", "HandE_(B)", "HandE_B4", "HandE_C3", "HandE_A4",
            "HandE_E1", "1_B_HE", "HandE_1", "HandE_FS_(DGD)", "1_A_HE", "HandE_2", "HandE_B5",
            "H_and_E_FSA", "H_and_E_D1", "HandE_(A)", "1_C_HE", "HandE_(B2)", "H_and_E_A3", "HandE_(A1)",
            "HandE_D2", "1_D_HE", "H_and_E_FS", "HandE_D", "H_and_E_B3", "H_and_E_2", "HandE_(A2)", 
            "HandE_C4", "HandE_D3", "HandE_4", "H_and_E_DGD", "H_and_E_C2", "HandE_A6", "HandE_E2",
            "HandE_(2)", "HandE_F1", "HandE_G1", "HandE_3", "H_and_E_E1", "HandE_B8", "HandE_B6",
            "HandE_B7", "HandE_(B1)", "HandE_(1)", "H_and_E_B", "HandE_AFS", "HandE_(B3)", "HandE_(C2)",
            "1_E_HE", "H_and_E_3", "HandE_(C1)", "H_and_E_A", "HandE_FSB", "HandE_FS_DGD", "HandE_E",
            "HandE_Td", "HandE_(FS)", "H_and_E_D2", "FS_H_and_E", "2_A_HE", "HandE_A5", "HandE_(3)", 
            "H_and_E_1", "H_and_E_C3", "HandE_A7", "HandE_BLOCK_B", "1_F_HE", "H_and_E_C4", "HandE_BLOCK_B1",
            "HandE_6", "HandE_F2", "HandE_BLOCK_C", "H_and_E_B4", "HandE_(C3)", "HandE_D4",
            "HandE_5", "Hande_FS1", "H_and_E_Tf", "HandE_(C4)", "HandE_G", "HandE_PBSa",
            "HandE_(C)", "HandE_BLOCK_A1", "HandE_C5", "HandE_(A3)", "HandE_E3", "H_and_E_7",
            "HandE_3B", "H_and_E_DGD", "HandE_B9", "H_and_E_A4", "HandE_Te", "H_and_E_C",
            "HandE_Tc", "HandE_(D)", "HandEB1", "HandE_(A5)", "HandE_C8", "H_and_E_Th", 
            "H_and_E_F3", "HandE_BLOCK_A3", "HandE_B11", "HandE_(8)", "H_and_E_FS1", "1_H_HE", 
            "1_G_HE", "HandE_DGD", "HandE_12", "HandE_7", "HandE_C6", "HandE_(B4)", "HandE_(FSB)",
            "HandE_(D2)", "H_and_E_B6", "HandE_H", "HandE_2D", "FS1_H_and_E", "H_and_E_5", "HandE_(B5)",
            "H_and_E_C8", "H_and_E_G1", "H_and_E_6", "HandE_FSC", "HandE_F", "HandE_(FSA)", "HandE_2A",
            "HandE_2C", "HandE_BFS", "HandE_3A", "H_and_E_14", "HandE_BLOCK_B3", "HandE_B14", "HandE_B22",
            "2_B_HE", "H_and_E_4", "HandE_Oa", "HandE_FS_A", "HandE_B13", "HandE_B10", "H_and_E_A8", "HandEB2",
            "H_and_E_BSa", "H_and_E_Syn", "HandE_(D5)", "HandE_CSF", "HandE_506915", "HandE_(2515_3)", 
            "HandE_(2431-B)", "_HandE_D1", "H_and_E_RSH1", "HandE_B-R", "HandE_504953", "HandE_504954", 
            "HandE_504952", "HandE_A11", "H_and_E_A1_7716", "H_and_E___C2", "H_and_E__C2", "HandE_504950",
            "HandE_504951", "HandE_504949", "DGD_H_and_E", "HandEB10", "H_and_E_2A", "4992-he-004", "956_2A_HE",
            "956_2A_HE-003", "666-he-003", "3423_2C_HE", "666-he", "3477-he-003", "10412_1D_HE", "9980_1B_HE",
            "956_HE", "3477-he-001", "1184-2A_19-HHE", "5432-HE", "956_2A_HE-001", "666-he-009", "666-he-002", 
            "3423_2D_HE", "3477-he-005", "4992-he", "3477-he-004", "1_I_HE", "4492-2B_HE", "3423_1A_HE",
            "10412_1A_HE", "956_1A_HE-003", "10412_1B_HE", "4992-he-001", "4492-HE", "3_A_HE", "10412_1E_HE", 
            "666-he-005", "4992-he-002", "3423_1A_HE-001", "666-he-007", "666-he-006", "666-imp_he", "1184-2A_HHE", 
            "666-he-004", "10412_1F_HE", "956_1A_HE-002", "3423_2E_HE", "3423-1A_HE", "4992-he-003", "1184-2A_20-HHE",
            "3477-he_(2)", "956_2A_HE-002", "666-imp_he-001", "HEE_C1", "4992-he-005", "3477-he", "956_1A_HE-001", 
            "3477-he-002", "666-he-001", "956_1A_HE", "Hand_E_A2", "Hand_E_B", "HandD_17", "Hand_D2", "hande", "HANDE", 
            "H_and_E", "4492-2A_H&E_Lvl_1", "HandE_506913", "HandE_2X", "HandE_(S14-904)", "HandE_PERM_of_FS_", "HandE_F_R",
            "HandE_2H", "HandE_fma", "HandE_tg", "HandE_30", "HandE_BLOCK_1", "HandE_2Z", "HandE_B2_", 
            "HandD_17", "HandE_FSA_DGD", "HandE_BLOCK_D1", "HandE_BR1", "H_and_E_22", "HandE_B1", "H_and_E_", "1_E_HE",
            "H_and_E_3", "H_and_E_A", "HandE_(B3)", "H_and_E_D2", "FS_H_and_E", "2_A_HE", "H_and_E_1", "H_and_E_C3",
            "1_F_HE", "H_and_E_C4", "H_and_E_B4", "H_and_E_DGD_", "HandE_9", "H_and_E_____C2", "HandE_504961", "HandE_504962",
            "HandE_504963", "H_and_E_frozen", "H_and_E_A-F", "H_and_E_Fa", "H_and_E_E3", "H_and_E__C3", "HandE_29", "HandE_27",
            "HandE_28", "HandE_506912", "HandE_504958", "H_and_E_B5", "H_and_E_B7", "HandE_B21", "HandE_B19", "HandE_H6", "HandE_I",
            "HandE4", "HandE_3D", "TP1_H_and_E", "H_and_E_D-3", "H_and_E_D-1", "HandE_BLOCK_C2", "HandE_F3", "HandE_AFS_biop",
            "HandE_initial_tumor", "HandE_C1_S-12-56", "HandE_E4", "HandE_H4", "HandE_BLOCK_A9", "HandE_BLOCK_C5", "HandEC",
            "HandE_FS4", "HandE_FS3", "4492-HE", "HandE_FSA_2", "H_and_E_G2", "HandE_Bsa", "H_and_E_9", "HandE_th", "HandE_A10",
            "HandE_S-15-3720", "HandE_S-05-6044", "HandE_B2_S-15-3720", "H_and_E__B-02", "H_and_E_B-02", "H_and_E_B-01", "HandE_D6",
            "HandE_2T", "HandE_2BB_2", "HandE_2J", "HandE_2BB_1", "HandE_2BB", "HandE_2P", "HandE_2I", "HandE_2Q", "HandE_(C5)",
            "HandE_(C7)", "HandE_A14", "HandE_506887", "HandE_CR1", "HandE_BLOCK_A", "HandE_BLOCK_B2", "HandE_BLOCK_C3",
            "HandE_(5)", "HandE_(10)", "HandE_(14)", "HandE_(17)", "HandE_506909", "HandE_FSB_1", "HandE_H_and_EA1", "HandE_Tf",
            "HandE_D9", "HandE_506921", "HandE_I_2", "HandE_G2", "HandE_J1", "H_and_E_Tb", "S-09-1537_HandE", "S-08-5417_HandE",
            "HandE_G3", "HandE_1FS", "HandE_2F", "HandE_(A8)", "H_and_E_A1-2", "H_and_E_A2-3", "HandE_504956", "HandE_504957",
            "HandE_504955", "HandE_(S-08-2219)", "HandE_(S-14-1113)", "HandE_A1_2937", "H_and_E_A5", "HandE_fb", "H_and_E_D",
            "H_and_E_D13", "H_and_E_FSB", "HandE_FSA1", "HandE_DR1", "3_A_HE", "H_and_E_AFS", "HandE_506917", "HandE_S-12-681",
            "HandE_S-06-2617", "H_and_E_A16", "H_and_E_A15", "H_and_E_A17", "H_and_E_A7", "H_and_E_C6", "H_and_E_B2-2", "H_and_E_B1-2",
            "HandE_B2_deeper_section", "H_and_E_A11", "H_and_E_A13", "H_and_E_C-1", "H_and_E_C-3", "HandE_504948", "HandE_504959",
            "HandE_504960", "HandE_PBS_b", "HandE_TF", "HandE_F_1", "HandE_506908", "HandE_PBS_6", "HandE(B1)", "HandE_506914",
            "HandE_AR2", "HandE_AR1", "HandE_(13)", "HandE_(S14-765)", "HandE_FSA3", "HandE_FSA2", "H_and_E_F-1", "H_and_E_J-1",
            "HandEB_1", "HandE_BRS1", "H_and_E_SFA", "HandE_CS-14-399", "H_and_E__B2", "HandE_BLOCK_A4", "HandE_BLOCK_A2",
            "HandE_504966", "HandE_504965", "HandE_504964", "HandE_BLOCK_2", "_HandE_FSA_B1", "HandE_A_Perm", "H_and_E_DFS01",
            "HandE_AR", "H_and_E_C9", "HandE_H_and_E_A_FS", "FS_HandE_", "H_and_E_F", "H_and_E_F1", "H_and_E_FS01", "HandE_B_10",
            "HandE_(E)", "HandE_(4)", "H_and_E_G1-2", "H_and_E__A1", "HandE_F4", "H_and_E_B9", "HandE_A_4", "HandE_506920",
            "H_and_E_C5", "H_and_E5", "10412_1E_HE", "10412_1B_HE", "10412_1A_HE", "HandE_FSB_Permanent", "H_and_EA1",
            "H_and_E_A_FS", "HandE_FSA_DGD_"],
    "EMA": ["EMA", "ema", "EMA_B1", "EMA_C1", "EMA_A1", "EMA_A2", "EMA_A", "EMA_B2", "EMA_B", "EMA_C2", "EMA_E1", "EMA_", "EMA_C3",
            "EMA_A4", "EMA_3_", "EMA_D2", "EMA_B5", "EMA_B3", "EMA_A5", "EMA_FSA", "3423_2C_EMA", "EMA_2", "EMA_E", "EMA_D3",
            "EMA_A3", "1_A_EMA", "EMA_C5", "EMA_D1", "EMA_B4", "EMA_B6", "EMA_(B2)", "EMA_C4", "1184-2A_EMA", "EMA_C", "EMA_C9",
            "EMA_A_", "10412_1B_EMA"],
    "GFAP": ["GFAP", "gfap", "GFAP_B1", "H_and_E_GFAP", "GFAP_A1", "GFAP_A2", "GFAP_C1", "GFAP_B2", "GFAP_B", "1_A_GFAP",
            "GFAP_B3", "1_B_GFAP", "GFAP_A", "GFAP_C2", "GFAP_A3", "GFAP_D1", "GFAP_C", "GFAP_A4", "GFAP_C3", "GFAP_D",
            "GFAP_D2", "GFAP_A5", "GFAP_FSA", "GFAP_FS", "GFAP_2", "GFAP_1", "GFAP_(A1)", "GFAP_B4", "GFAP,_GFAP",
            "1_D_GFAP", "1_C_GFAP", "GFAP_(C4)", "GFAP_(D5)", "GFAP_3", "GFAP_(8)", "GFAP_E3", "GFAP_D3", "9980_1B_GFAP",
            "4492-GFAP", "3423_2C_GFAP", "GFAP_(C2)", "GFAP_(A2)", "GFAP_(C3)", "GFAP_SS-15-3720", "GFAP_S-05-6044", "GFAP_B2_S-15-3720",
            "GFAP_(B)", "666-gfap-001", "666-gfap", "GFAP_A7", "GFAP_G1", "GFAP_BLOCK_A3", "GFAP_BLOCK_B1", "2_B_GFAP", "2_A_GFAP",
            "4492-1E_GFAP", "4492-1F_GFAP", "4492-1A_GFAP", "GFAP_B9", "5432-GFAP", "GFAP_(C1)", "GFAP_(3)", "GFAP_B6", "3477-gfap-001",
            "3477-gfap", "1_E_GFAP", "956_1A_GFAP", "1184-2A_GFAP", "GFAP_(B1)", "GFAP_(S-14-904)", "GFAP_FSB", "GFAP_(A)", "GFAP_(B2)",
            "GFAP_4", "GFAP_BLOCK_2", "GFAP_BLOCK_1", "10412_1B_GFAP", "GFAP_E1", "GFAp", "GFAP__C2"],
    "KI-67": ["KI-67", "ki-67", "Ki-67", "KI67", "Ki067_A2", "Ki-62", "Ki-57", "Ki67", "Ki-67_B1", "H_and_E_Ki-67", "KI-67_A1", "KI-67_A2",
            "KI-67_C1", "KI-67_B", "KI-67_B2", "1_A_Ki67", "1_B_Ki67", "KI-67_B3", "ki-67_C2", "ki-67_D1", "KI-67_A3", "KI-67_C", "KI-67_",
            "ki-67_A4", "KI-67_A", "ki-67_(B)", "Ki-67_C3", "1_D_Ki67", "Ki-67_D", "Ki-67_2", "Ki-67_A5", "Ki-67_3", "Ki67_A1", "Ki067_A2",
            "Ki67_C2", "Ki-67_D2", "1_C_Ki67", "Ki-67_FSA", "Ki-67_1", "Ki-67_FS", "Ki-67_C4", "Ki-57", "2_A_Ki67", "Ki-67_(B2)", "Ki67_(D5)",
            "Ki-67_E1", "Ki-67__C2", "Ki-67_(2)", "Ki67_B1", "Ki-62", "1_F_Ki67", "Ki67_B3", "__Ki-67_D1", "9980_1B_KI67", "KI-67_BLOCK_C",
            "Ki-67_BLOCK_D", "4492-Ki67", "Ki-67_(C2)", "KI-67_B4", "Ki-67_(A2)", "Ki-67_(C3)", "Ki-67_A10", "Ki-67_S-05-6044", "Ki-67_B2",
            "666-ki67-001", "666-ki67", "KI67_BLOCK_A3", "KI67-_BLOCK_B1", "Ki-67_E", "2_B_Ki67", "4992-ki67", "4992-ki67-001", "4992-ki67-002",
            "KI67_C1", "Ki-67,_Ki-67", "Ki-67_B9", "5432-Ki67", "Ki-67_(S-08-2219)", "_Ki-67_B1", "Ki-67_(3)", "Ki67_B2", "Ki-67_(FS)", "Ki-67_B4",
            "Ki-67_B6", "3477-ki67", "3477-ki67-001", "Ki-67-A", "1_E_Ki67", "956_1A_KI67", "1184_-_2A_Ki67_MIB-1", "Ki-67_(S-14-904)", "Ki-67_FSB",
            "Ki-67_(D)", "Ki-67_(A)", "KI-67_BLOCK_2", "KI-67_BLOCK_1", "Ki-67_C9", "Ki-67_(A1)", "10412_1B_KI67"],
    "INI-1": ["INI-1", "ini-1", "INI_A2", "INI_1_B1", "INI_1_B2", "INI_1_C1", "INI1", "INI_C", "INI", "INI-1_B1", "INI-1_A1",
            "1_B_INI-1", "INI-1_A", "INI-1_", "INI-1_B3", "INI-1_D1", "INI-1_C4", "INI-1_B", "INI-1_B2", "INI-1_C1", "INI-1_E1",
            "INI-1_B8", "INI-1_B5", "INI-1_(SMARCB1)", "INI-1_A2", "INI-1_D2", "INI-1_C3", "INI-1_B9", "1_A_INI-1", "INI-1_D",
            "1_E_INI-1", "INI-1_A3"],
    "SMA": ["SMA", "sma", "Sma", "SMA_A2", "SMACT", "SMACT_A2", "SMACTIN_BLOCK_B1", "SMACTIN_BLOCK_A3", "SMACTIN_B1", "SMACT_B",
            "SMACT_D", "SMACT_A", "SMACTIN"],
    "SYNAPTO": ["Synapto", "SYNAPTO", "Syn", "syn", "SYN_D", "SYN_3", "SYN_A", "Syn_B3", "Syn_A", "SYN",
                "Syn_B2", "Syn_B1", "Syn_A2", "SYN_A1", "SYN_B1", "SYN_", "SYN_B", "SYN_A2", "SYN_(A2)", "SYN_B2", "SYN_C1",
                "SYN_4", "Synaptophysin", "1_B_Synaptophysin", "1_A_Synaptophysin", "Synaptophysin_A1", "Synaptophysin_B2", "Synaptophysin_(D5)",
                "Synaptophysin_(C4)", "1_F_Synaptophysin", "4492-Synapto", "Synaptophysin_C1", "3423_2C_SYNAPTO", "synaptophysin_A1", "Synapto_A7",
                "Synapto_A2", "4992-synapto-001", "4992-synapto", "4992-synapto-002", "1_D_Synaptophysin", "1_C_Synaptophysin", "3477-synapto",
                "3477-synapto-001", "1_E_Synaptophysin", "956_1A_SYNAPTO", "1184-2A_Synapto", "Synapto_B", "Synapto_A1", "Synapto_A4", "10412_1B_SYNAPTO"],
    "BRG1": ["BRG-1", "BRG1", "brg-1", "brg1", "BRG_1", "BRG_B1", "BRG1_E1", "1_B_BRG1", "BRG-1_C4", "BRG-1_B2", "BRG-1_(SMARCA4)", "BRG-1_A1"],
    "RETICULIN": ["RETICULIN", "reticulin", "RETIC_C3", "RETIC_B1", "RETIC_B1", "Ret_B2", "RET", "Ret", "RETIC_B2", "Ret_D2", "Ret_B3",
                "RETIC", "Ret_B3", "RETIC_C5", "Ret_D3", "RETIC_A3", "3477-retic", "Ret_D2", "Ret_B2", "Ret_D3", "Reticulin_FSA", 
                "Reticulin_B9", "Reticulin_B3", "Reticulin_B2", "Reticulin_C1", "3477-retic", "3477-retic-001", "Reticulin_(B)", "Reticulin_FS",
                "Reticulin_A1", "Reticulin_A2", "Reticulin_", "Reticulin_D1", "Reticulin_B1", "1_B_Reticulin", "1_A_Reticulin", "1_D_Reticulin",
                "1_E_Reticulin", "Reticulin_A", "Reticulin_B", "Reticulin_A3", ""],
    "VIMENTIN": ["VIMENTIN", "vimentin", "Vimentin", "Vim_", "vim_", "VIM", "Vimentin_A2", "Vimentin_B1", "Vimentin_B3", "Vimentin_B2", "Vimentin_3",
                "Vimentin_C1", "Vimentin_B", "Vim_A2", "Vimentin_A3", "Vimentin_B4", "Vimentin_B5", "VIM_A2", "VIMENTIN-_BLOCK_B1", "VIMENTIN-_B1",
                "Vimentin_B9", "Vimentin_A", "Vimentin_C4", ""],
    "OLIG2": ["OLIG-2", "Olig-2", "1184-2A_OLIG2_Control", "1_A_OLIG2", "1_C_OLIG2", "OLIG_2", "OLIg-A3", "1184-2A_16-OLIG2",
            "Olig_2", "1_E_OLIG2", "Oligo-1_B1", "956_1A_OLIG2_CTRL", "OLIG_2___C2", "Oligo-2_B1", "1_B_OLIG2", "Oligo-2",
            "OLIG_C1", "Olig2", "Olig_-2_C1", "10412_1B_OLIG2-001", "956_1A_OLIG2", "OLIGO-2", "OLIG-2_B1", "Olig-2_A2",
            "Olig-2_B1", "_OLIG-2_A1", "OLIG-2_A1", "OLIG-2_B4", "Olig-2_A1", "OLIG-2_A3", "OLIG-2_A2", "Olig-2_B3",
            "OLOG-2"],
    "TTR": ["TTR", "TTR_A2", "TTR_A1", "TTR_A5", "TTR_B3", "956_1A_ATTR_CTRL", "1184-2A_ATTR_Prealb", "956_1A_ATTR", "1184-2A_ATTR", "956_1A_ATTR",
            ""],
    "CATENIN_BETA-1": ["beta-CAT", "Beta-catenin", "beta_catenin_B1", "beta_catenin", "Beta-Catenin_B1", "Beta_catenin_C1",
                        "Beta_-catenin", "Beta-catenin_B2", "Beta_catenin", "Beta-CAT", "beta_catenin_A2", "Beta_catenin_A1",
                        "beta-catenin", "Beta_CAT", "Beta-Cat", "Beta_catenin_B2", "Beta_Catenin", "beta-catenin_A2", "Beta_cat",
                        "Beta-CAT_A2", "Beta-Catenin_", "Beta-Catenin", "Beta_Catenin_B1", "Beta-catenin_B1", "1_B_B-catenin",
                        "1_A_B-catenin", "1_E_B-catenin", "1184-2A_B-catenin", "956_1A_BCATENIN", "B-catenin", "B-CAT", 
                        "B-cat", "Beta-catinin"],
    "P53": ["2_B_p53", "P53_D1", "p53_C3", "P53_C2", "P53_(+)", "p53", "P53_B1", "p53_B1", "2_A_p53", "P53_B4",
            "1184-2A_P53", "P53", "10412_1B_P53", "p53_C1", "p53_(3)", "1_A_p53", "p53_B3", "P53_A1", "P53_A2",
            "p53_A1", "956_1A_P53", "P53_A4", "p53_A3", "1_D_p53", "p53_A4", "P53A1", "P53_C1", "1184-2A_P53-001",
            "1_B_p53", "p53_C2", "P53_B2", "p53_A2", "1_E_p53", "10412_1B_P53"],
    "DESMIN": ["Desmin(_A)", "DES_B2", "Desmin_C1", "Desmin", "DES_B1"],
    "OCTA4": ["OCTA4", "OCTA_4_A1", "OCTA4_B1", "Octa-4", "OCTA_4", "OCTA-4_C", "OCTA-4_B2", "OCTA-4", "Octa_4",
            "OCTA-4_", "OCTA-4_B1", "OCTA_4_A2", "OCTA4_A7", "OCTA_4_B1", "Octa_4_", "OCTA_-4", "OCT_-4_B"],
    # cytokeratin AE1 AE3
    "CK_AE1_AE3": ["CYTOKERATIN", "Cytokeratin_C1", "cytokeratin_A5", "cytokeratin_A1", "Cytokeratin_AE1-3", 
                            "Cytokeratin", "AE1_AE3_3", "AE1_AE3_A", "AE1-AE3", "AE1_B", "Keratin_AE1_AE3_(A)", "AE1_AE3_B2", 
                            "CK_AE1_AE3", "AE1_AE3_B3", "AE1_AE3", "AE1_B4", "AE1_AE3_2", "AE1_AE3_A2", "AE1_AE3_A2_",
                            "AE1_B1", "AE1_AE3_B9", "AE1_C", "AE1_AE3_C1", "AE1_AE3_D", "AE1_B2"],
    "TRIMETHYL": ["trimethyl-3_B1", "Trimethyl_H3_A1", "Trimethyl_H3", "Trimethyl", "Trimethyl_A1", "TrimethylH3",
                "Trimethyl_3", "Trimeth_3", "trimethyl_B1", "trimethyl_3_C4", "trimethyl_H3", "Trimeth_3_A2", "Trimethyl_3_D1",
                "Trimethyl_C1", "trimethyl_B2", "trimethyl_3_B1", "trymetyl_C2", "Trymetyl_B1", "trymethyl", "Trymethyl_A1",
                "Trymethyl_B1", "Trimethyl_H3_A1", ""],
    "TRICHROME": ["trichrome_elastic", "TRICHROME", "TRI_A3", "Trichrome_A", "Trichrome", "TRI_B1", "trichrome_C2",
                 "Trichrome_D", "M-TRI", "TRI_B2"],
    "IDH1": ["2_A_IDH1", "IDH1_C3", "IDH-1_B2", "IDH-1_B1", "IDH-1_B3", "IDH_-1_C1", "IDH-1", "1_B_IDH1", "IDH-1_A3",
            "IDH-1_A2", "1_D_IDH1", "IDH-1_A4", "IDH1_A1", "IDH-1_A1", "IDH1", "666-idh1", "IDH1_(-)", "IDH-1_B4"],
    "ALCIAN_BLUE": ["Alcian_blue", "Alcian_blue_B1", "Alcian_blue_FSA", "Alcian_blue_A4", "Alcian_Blue", "alcian_blue"],
    "HISTONE_H3": ["Histone_3_2937", "Histone_3_A5", "Histone_3_A1", "Histone_3_B1", "Histone3_B3", "Histone_3_D1", "Histone_3",
                "Histone_K27M", "Histone_3_B2", "Histone_3_C1", "Histone_3_C2", "Histone_3_C3", "Histone3_B1", 
                "Histone_3_", "Histone_3_C4", "Histone_3_A2", "Histone_3", "Histone3_C2", "Histone_-3", "Histone_3,_Histone_3"],
    "LUXOL_FAST_BLUE": ["Luxol_Fast_blue", "Luxol_fast_blue_A2", "Luxol_fast_blue_C1", "Luxol", "Luxol_fast_blue", "Luxol_A4",
                        "LFB_(C3)", "LFB_(A2)", "LFB_B1"],
    "LANGERIN": ["Langerin_", "langerin", "Langerin_B3", "Langerin", "Langherin"],
    "NEUROFILAMENT": ["1_B_Neurofilament", "Neurofilamen", "2_A_Neurofilament", "Neurofilament", "neurofilament", "NF", "NF_2F11_A4",
                    "NF_A1", "NF_2F11", "NF_C5","NFP", "nfp", "NFP_A1", "NFP_B1", "NFP_A2", "NFP_B2", "NFP_C1", "NFP_B", "NFP_B3", "NFP_A3", 
                    "NFP_A", "NFP_C2", "NFP_D", "NFP_3", "NFP_D2", "NFP_2", "NFP_1", "NFP_(A2)", "NFP_(B)", "NFP_D1", "NFP_(D5)", "NFP_(C4)", 
                    "NFP_C", "9980_1B_NFP", "NFP_FS", "NPF_A3", "NFP_(C2)", "NFP_(A1)", "NFP_(C3)", "NFP_(A)", "NFP-_BLOCK_B1", "NFP-_BLOCK_A3", 
                    "NFP_-B3", "NFP_B9", "5432-NFP", "NFP_B4", "NFP_C3", "NFP2F11_B1", "NFP_A4", "NFP_(B2)", "NFP__A1", "NFP_C5"],
    "INHIBIN": ["Inhibin_B1", "Inhibin_B", "3423_2C_INHIBIN", "Inhibin_A3", "Inhibin"],
    "PROLACTIN": ["Prolactin", "Prolactin_B1", "prolactin", "PRL"],
    "FILAMIN": ["Filamin_B2", "filamin", "Filamin_A", "Filamin", "Filamin_-A", "Filamin-1", "Filamin_B1", "Filamin-A", "Fimamin_A1",
                "Filamin"],
    "BRACHYURY": ["3423_2C_BRACHYURY_CTRL", "3423_2C_BRACKYURY", "Brachyury", "brachyury", "Brach_A1", "3423_2C_BRACHYURY_CTRL"],
    "CHROMOGRANIN": ["chromogranin", "CHROMO", "Chromogranin", "Chrom", "1184-2A_Chromograni", "Cromogranin"], 
    "NEUN": ["1_F_NEUN", "NEU-N_C3", "3477-neun", "NEU-N_B1", "1_I_NEUN", "NEU-N", "NEU-N_A3", "Neu-N_B1", "NEU-N_C1",
            "NEUN_B3", "NEUN_A", "NEU_A2", "NEU-N_(1)", "NEU-N_B2", "NEUN_4", "NEU-N_(A2)", "NEU_-N__A1", "NEU-N_B2_S-15-3720",
            "NEU-N_D2", "1_C_NEUN", "Neu-N_C1", "NEU-N_B3", "NEUN_B2", "NEU-N_FSA", "NEU-N_B", "NEU-N_C2", "NEU-N_D1",
            "1_B_NEUN", "NEU-N_D3", "NEUN_B1", "NEU_-N_A3", "NEU_N", "NEU-M", "NEU-N_A1", "3477-neun-001", "NEU-N_A2", "NEU-N_(C3)",
            "NEU-N_(2)", "NUE-N_B3", "NWU-N", "NGU-N"],
    "NF2": ["NF2N11_4", "NF2FII_A3", "NF2", "NF2F11_", "NF211", "NF2F11"],
    "ATRX": ["ATRX_C3", "ATRX", "ATRX_G1", "ATRX_B1", "2_B_ATRX", "ATRX_A1", "666-atrx", "1_A_ATRX", "ATRX_A2", "1_D_ATRX",
            "ATRX_C1", "ATRX_B2", "ATRX_B4", "2_A_ATRX", "1_B_ATRX", "ATRX"],
    "TOL_BLUE": ["Tol_Blue_SPA", "Tol_Blue_SP"],
    "S100": ["S100_B3", "2_A_S100", "S100", "S-100_D3", "S100_B1", "S100_C5", "S-100_B3", "S100_FS", "S-100_D2",
            "S-100_A1", "S-100_B4", "1_B_S100", "1184-2A_S100", "9980_1B_S100", "S100_B2", "S-100_B21", "S100_C1",
            "S100_FSB", "S-100_A2", "S100_B", "S-100_B2", "S-100_D1", "sS-100", "S-100_A4", "S100_A3", "S-100_C1",
            "S-100_B", "S100_A", "S-100_B1", "S100_A1", "S-100_C2", "S-100", "s-100"],
    "LEF1": ["LEF_B2", "1_B_LEF-1", "LEF_A2", "LEF_-1_B2", "LEF1", "1_A_LEF-1", "LEF-1", "LEF_1", "1_E_LEF-1", "LEF", "LEF_B1"],
    "BRAF": ["1_I_BRAF", "1_C_BRAF", "1_D_BRAF", "1_A_BRAF", "1_B_BRAF"],
    "PHH3": ["2_A_PHH3", "1_B_PHH3", "1184-2A_PHH3", "1_F_PHH3", "1_A_PHH3", "1184-2A_PHH3-001", "1_C_PHH3"],
    "GAB": ["GAB-1", "GAB_B2", "GAB_-1", "GAB-A2", "GAB1", "GAB-1_B2", "GAB-1_A2", "GAB_B1", "GAB-1_B1", "GAB"],
    "MYOGENIN": ["MYOGENIN", "MYOG_B2", "MYOD1", "Myogenin", "MYOG_B1"],
    "PAS": ["PAS_FSA", "PAS_C", "PAS_B", "1_A_PAS", "PAS_B1", "PAS_C1", "1_B_PAS", "1_A_PAS-D", "PAS", "1_B_PAS-D", "PAS"],
    "SOX10": ["1_A_SOX-10", "1_B_SOX-10", "9980_1B_SOX10_CTRL", "SOX-10_C1", "1_D_SOX-10", "1_C_SOX-10", "9980_1B_SOX10", "SOX10", 
            "2_A_SOX-10"],
    "CD34": ["CD34", "CD34_A2", "CD34_A1", "CD34_B1", "CD34_C1", "CD34_A3", "1_B_CD34", "CD34_B2", "CD34_A4", "CD_34_A1_7716", "CD34_C2",
            "CD34_D2", "1_A_CD34", "CD34_BLOCK_A3", "CD34_B3", "CD34_FSB", "CD34_C9", "CD34_D1", "CD_34_B2", "CD_34_B1", "CD_34"],
    "YAP": ["YAP", "YAP-1", "YAP_", "YAP_B1", "YAP_B2", "YAP_1", "YAP_A2", "Yap"],
    "PLAP": ["PLAP", "PLAP_B2", "PLAP_A1", "PLAP_A7", "PLAP_4", "PLAP_B", "PLAP_C", "PLAP_"],
    "KLUVER": ["KLOVER_B", "KLOVER_B2", "KLOVER_B1", "Kluver"],
    "CD56": ["CD56", "CD56_A2", "1184-2A_CD56", "956_1A_CD56", "956_1A_CD56", "CD56_A2", "1184-2A_CD56"],
    # O13, MIC2, CD99 are the same 
    "CD99": ["CD99", "MIC2_A1", "MIC-2", "MIC2_", "MIC2", "MIC_2", "O13", "O13_A1", "o13", "CD_99_(013)", "CD99_MIC2", "CD99_013",
            "013_CD99", "O13_A1", "O13_CD99"],
    "CD45": ["CD45", "CD45_A1", "CD-45_B2", "CD-45_B1", "CD45_(A2)", "CD45_(C3)", "CD45_4", "CD45_B1", "CD_45_B2", "CD_45_FSB",
            "CD_45_B1", "CD45_C2", "CD45_C5"],
    "CD117": ["CD117", "CD117_A1", "Cd117_B2", "CD-117", "CD117_A2", "CD_117_4", "CD117_B1", "CD117_C", "CD117_B", "CD_117",
            "CD-117_B1", "C-Kit"],
    "FACTOR8": ["F8_B2", "F8_B3", "F8_B9", "F8", "F_VIII"],
    "ALK1": ["ALK1_BLOCK_A3", "ALK1-_BLOCK_B1", "ALK1_B3", "ALK"],
    "NKX2.2": ["NKX2.2", "NKX2.2_B1", "NKX2.2_A1"],
    # cytokeratin 7
    "CK7": ["CK7", "CK7_A2", "CK7_A1", "CK7_B3", "1184-2A_CK7", "956_1A_CK7", "CK7_B2", "CK_7_B4"],
    "CD163": ["CD-163", "CD163_B1", "CD163_A1", "CD_163_B1", "CD_163_B2", "CD_163_FSB", "CD163_C2", "CD163"],
    "CD1A": ["CD1a", "CD1A"],
    "PHOX2B": ["PHOX2B", "PHOX2B_A1", "PHOX2B_B6"],
    "CD20": ["CD20", "CD20_F1"],
    "CD3": ["CD3", "CD3_A2", "CD3_F1"],
    "CD10": ["CD10", "3423_2C_CD10"],
    "AFP": ["AFP", "AFP_C", "AFP_4"],
    # cytokeratin 20
    "CK20": ["CK20", "CK20_B3", "1184-2A_CK20", "1184-2A_CK20_Control"],
    "CD68": ["CD68_B1", "CD68_B2", "4492-CD68", "3423_2C_CD68KP1", "CD68_C1", "CD68_A2", "CD68_FSA3", "CD68", "CD_68_B1",
            "CD_68_FSB", "CD_68_B2"],
    "HCG": ["HCG_C", "HCG_B", "HCG_A7", "HCG_4", "HCG"],
    "FACTOR13": ["Factor_XIII_a", "Factor_XIII", "3423_2C_FACTOR_XIII"],
    "SALL4": ["1_C_SALL4", "1_B_SALL4", "1_A_SALL4", "SALL4_A2", "Sall4"],
    "SSTR2": ["SSTR2", "SSTR2_C9", "SSTR2_B2"],
    "C3L": ["C3L-05726-25", "C3L-05726-24", "C3L-05726-23", "C3L-05726-22", "C3L-05726-21"],
    "RA21": ["RA21_Th", "Ra21_F6"],
    "ACTH": ["ACTH"],
    "TH": ["TH"],
    "GH": ["GH"],
    "LH": ["LH"],
    "EGFR": ["1_B_EGFR"],
    "CD5": ["CD5"],
    "CD79": ["CD79"],
    "FSH": ["FSH"],
    "CD30": ["CD30", "CD30_A7"],
    "DGD": ["FSA_DGD", "FSA_DGD_"],
    "AAT": ["AAT"],
    "Lys": ["Lys"],
    "IDN1": ["IDN1"],
    "NPF": ["NPF"],
    "TTF": ["TTF"],
    "CD31": ["CD31_B21", "CD31"],
    "PGP": ["PGP_B4", "PGP"],
    "CD57": ["CD-57_B2", "CD-57_B1"],
    "COL4": ["COL_4_B1", "COL_4_B2"],
    "Glypican": ["Glypican_3"],
    "TDT": ["TDT"],
    "STAT6": ["STAT_6"],
    "NSE": ["3423_2C_NSE_CTRL", "3423_2C_NSE", "NSE_B2"],
    "T14": ["T14"],
    "SF1": ["SF1"], 
    "GATA3": ["GATA3"],
    "BCOR": ["1_B_BCoR"],
    "NFR": ["NFR_A2"],
    "MAP2": ["MAP2_BLOCK_B1", "MAP2_-BLOCK_A3"],
    "SATB2": ["SATB2", "SATB2_A2"],
    "LMWK": ["LMWK_C", "LMWK_B"],
    "NB84": ["NB84"],
    "CFAP": ["CFAP_C5", "CFAP_C1"],
    "TOXO": ["TOXO"],
    "IGG4": ["IgG4"],
    "AFB": ["AFB_E1", "AFB"],
    "IGG": ["IgG"],
    "CD4": ["CD4"],
    "GRAM": ["GRAM"],
    "CD8": ["CD8"],
    "FITE": ["FITES"],
    "SV40": ["SV40"],
    "GMS": ["GMS"],
    "P52": ["P52"],
    "CAM5.2": ["CAM_5.2", "956_1A_CAM5.2", "1184-2A_CK_CAM_5.2", "CAM5.2"],
    "PAN-KERATIN": ["1184-2A_Pan_Keratin"],
    "PinA": ["956-imp_pina"],
    "CDH1": ["956_1A_ECAD", "956_1A_ECAD_CTRL", "1184-2A_ECAD", "1184-2A_ECAD-001"],
    "HER2": ["Her2_Neu_(-)"],
    "GLUT1": ["GLUT-1"],
    "WT1": ["WT1"],
    "IRON": ["Iron_stain_B1"],
    "HMWK": ["HMWK_B", "HMWK_C"],
    "ORO": ["ORO", "Oil-red-O"],
    "MELA": ["MELA"],
    "HMB45": ["HMB45"],
    "FONTANA": ["Fontana_A2"],
    "AIF1": ["666-iba1"],
    "EZHIP": ["EZHIP_B1"],
    "CD42": ["CD42B"],
    "CD33": ["CD33"],
    "SPIRO": ["SPIRO"],
    "CD19": ["CD19"]
}

def explore(path):
    stain_counts = {}
    
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.svs'):
            for stain, stain_names in stains.items():
                file_stain = os.path.splitext(filename)[0] 
                for name in stain_names:
                    if name.lower() == file_stain.lower():
                        file_stain = stain
                        break
                else:
                    continue
                break
            if file_stain is not None:
                stain_counts[file_stain] = stain_counts.get(file_stain, 0) + 1
            else:
                stain_counts[filename] = stain_counts.get(filename, 0) + 1
    
    df = pd.DataFrame(stain_counts.items(), columns=['Stain Methods', 'Count'])
    return df

explore_df = explore(histology_path)

output_path = '/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/SUMMARY_FILES/HISTOLOGY'
csv_file_path = os.path.join(output_path, 'stain_summary.csv')
explore_df.to_csv(csv_file_path, index=False)

# %% BRAIN TUMOUR TYPES CSV 
# load data
df = pd.read_csv(r'/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/SUMMARY_FILES/HISTOLOGY/CBTN_histology_summary.csv')

# remove 'Not available' subjectID
df_tumor_types = df.loc[df['subjectID'] != 'Not_available']

df_tumor_types = df_tumor_types[['subjectID', 'diagnosis']]

# unique combination of 'subjectID' and 'diagnosis'
df_tumor_types = df_tumor_types.groupby(['subjectID', 'diagnosis']).first().reset_index()

sorted_tumor_types = df_tumor_types['diagnosis'].value_counts().sort_values(ascending=False)

df_diagnosis_sums = pd.DataFrame(sorted_tumor_types).reset_index()
df_diagnosis_sums.columns = ['Diagnosis', 'Count']

csv_file_path = "/run/media/chrsp39/CBNT_v2/Datasets/CBTN_v2/SUMMARY_FILES/HISTOLOGY/tumour_types.csv"
df_diagnosis_sums.to_csv(csv_file_path, index=False)
# %%
