# %%
import os
from functools import partial
import timm
from .timm_wrapper import TimmCNNEncoder
import torch
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms
from .hipt_model_utils import get_vit256
from models.open_clip_custom import create_model_from_pretrained
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from transformers import AutoModel 
from models.resnet_custom_dep import resnet50_baseline

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
def get_encoder(model_name, target_img_size=224):
    print('loading model checkpoint')
    if model_name == 'resnet50' or model_name == 'resnet50_baseline':
        model = resnet50_baseline(pretrained=True)
        model = model.to(device)
    elif model_name == 'uni':
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load("/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/feature_encoders/UNI/pytorch_model.bin", map_location=device), strict=True)
        model = model.to(device)
    elif model_name == 'uni2-h':
        timm_kwargs = {
            'model_name': 'vit_giant_patch14_224',
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
            }
        model = timm.create_model(pretrained=False, **timm_kwargs)
        model.load_state_dict(torch.load(os.path.join("/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/feature_encoders/UNI2-h", "pytorch_model.bin"), map_location="cpu"), strict=True)
        model = model.to(device)
    elif model_name == 'conch_v1':
        model, _ = create_model_from_pretrained("conch_ViT-B-16", '/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/feature_encoders/CONCH_v1/pytorch_model.bin')
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        model = model.to(device)
    elif model_name == 'conch_v1_5':
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
        assert target_img_size == 224, 'TITAN is used with 448x448 CONCH v1.5 features'
    elif model_name == 'hipt':
        model = get_vit256(pretrained_weights='/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/feature_encoders/HIPT/vit256_small_dino.pth').to(device)
    elif model_name == 'prov-gigapath':
        model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=False)
        model.load_state_dict(torch.load("/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/feature_encoders/PROV-GIGAPATH/pytorch_model.bin", map_location=device), strict=True)
        model = model.to(device)    
    elif model_name == 'virchow':
        model = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.load_state_dict(torch.load("/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/feature_encoders/VIRCHOW/pytorch_model.bin", map_location=device), strict=True)
    elif model_name == 'virchow2':
        model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=False, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        model.load_state_dict(torch.load("/local/data1/chrsp39/CBTN_Histology_Multi_Stain/models/feature_encoders/VIRCHOW2/pytorch_model.bin", map_location=device), strict=True)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    print(model)
    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(mean=constants['mean'],
                                         std=constants['std'],
                                         target_img_size = target_img_size)

    return model, img_transforms