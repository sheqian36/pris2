# from . import datasets
# from . import engine_for_colorization
# from . import modeling_colorization
# from . import modeling_finetune
# from . import optim_factory
# from . import run_colorization
from . import utils
from . import test


# import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path
from collections import OrderedDict

from timm.models import create_model
from timm.utils import ModelEma
from torchvision import datasets, transforms
import pris2.modeling_colorization
from pris2.utils import NativeScalerWithGradNormCount as NativeScaler
from pris2.utils import *
from PIL import Image

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
# random.seed(seed)
cudnn.benchmark = True


def load_model(checkpoint_path, device):

    assert os.path.exists(checkpoint_path), "file: '{}' dose not exist.".format(checkpoint)
    model = create_model(
        "colorization_vit_large_patch16_224_fusion_whole_up",
        pretrained=False,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )

    patch_size = model.encoder.patch_embed.patch_size
    window_size = (224 // patch_size[0], 224 // patch_size[1])

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    checkpoint_model = None
    model_key = "model|module"
    for model_key in model_key.split('|'): # default='model|module'
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    if checkpoint_path.startswith('pretrain'):
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                # new_dict[key[8:]] = checkpoint_model[key]
                new_dict[key] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
    else:
        for key in all_keys:
            if key.startswith('patch_embed.'):
                new_dict['encoder.'+ key] = checkpoint_model[key]
            elif key.startswith('pos_embed'):
                new_dict['encoder.'+key] = checkpoint_model[key]
            elif key.startswith('blocks.'):
                new_dict['encoder.'+key] = checkpoint_model[key]
            elif key.startswith('norm.'):
                new_dict['encoder.'+key] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    load_state_dict(model, checkpoint_model, prefix="")
    model.to(device)

    return model



def load_img(img_path):

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    img = Image.open(img_path).convert('RGB')
    img = data_transform(img)

    return img


def color(img_path, caption, output_path, checkpoint_path="/home/shuchenweng/cz/oyh/data/pris/pris2/largedecoder.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("start load model")
    model = load_model(checkpoint_path, device)
    print("load model finished")
    model.eval()

    image = load_img(img_path)
    image = image.unsqueeze(0).to(device, non_blocking=True)
    # image = image.to(device, non_blocking=True)
    color_data = get_colorization_data(image)
    img_l = color_data['A'] # [-1,1]
    img_ab = color_data['B'] # [-1,1]
    # compute output
    with torch.cuda.amp.autocast():
        output, occm_pred = model(img_l.repeat(1,3,1,1), [caption])
    img_ab_fake = output

    fake_rgb_tensors = lab2rgb(torch.cat((img_l, img_ab_fake), dim=1))
    fake_rgbs = tensor2im(fake_rgb_tensors)
    save_img_fake = Image.fromarray(fake_rgbs[0])
    save_img_fake.save(output_path)

    print("colorization finished")