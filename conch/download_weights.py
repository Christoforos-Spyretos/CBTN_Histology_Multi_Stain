# %%
# from conch.open_clip_custom import create_model_from_pretrained
# model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="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")
# model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "CONCH/checkpoints/conch")

# %%
import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

login('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')  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "/home/chrsp39/CBTN_Histology_Multi_Modal/models/CONCH/checkpoints/conch"
os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
hf_hub_download("MahmoodLab/conch", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)

# %%