# %%
import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from HIPT_4K.hipt_model_utils import get_vit256
from models.open_clip_custom import create_model_from_pretrained
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50':
        model = TimmCNNEncoder()
    elif model_name == 'uni_vit':
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load("/home/chrsp39/CBTN_Histology_Multi_Modal/models/UNI/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin", map_location=device), strict=True)
        model = model.to(device)
    elif model_name == 'conch_vit':
        model, _ = create_model_from_pretrained("conch_ViT-B-16", '/home/chrsp39/CBTN_Histology_Multi_Modal/models/CONCH/checkpoints/conch/pytorch_model.bin')
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        model = model.to(device)
    elif model_name == 'hipt_vit':
        model = get_vit256(pretrained_weights='/home/chrsp39/Cross_modal_data_fusion/models/CLAM/HIPT_4K/Checkpoints/vit256_small_dino.pth').to(device)
    elif model_name == 'prov-gigapath':
        model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=False)
        model.load_state_dict(torch.load("/home/chrsp39/CBTN_Histology_Multi_Modal/models/PROV-GIGAPATH/feature_encoder/prov_gigapath/pytorch_model.bin", map_location=device), strict=True)
        model = model.to(device)    
    elif model_name == 'virchow':
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.load_state_dict(torch.load("/home/chrsp39/CBTN_Histology_Multi_Modal/models/VIRCHOW/feature_encoder/virchow/pytorch_model.bin", map_location=device), strict=True)
    elif model_name == 'virchow2':
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.load_state_dict(torch.load("/home/chrsp39/CBTN_Histology_Multi_Modal/models/VIRCHOW2/feature_encoder/virchow2/pytorch_model.bin", map_location=device), strict=True)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms