# %% IMPORTS
import torch 
from HIPT_4K.hipt_model_utils import get_vit256

# %%
ckpt_path = '/home/chrsp39/Cross_modal_data_fusion/models/CLAM/HIPT_4K/Checkpoints/vit256_small_dino.pth'
device256 = torch.device('cpu')

def vit_256():
    model = get_vit256(pretrained_weights=ckpt_path)
    return model