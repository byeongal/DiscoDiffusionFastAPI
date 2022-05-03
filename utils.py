from typing import List, Dict

import torch
import clip
import cv2
import torchvision.transforms as T
from loguru import logger

from dependencies.guided_diffusion.guided_diffusion.script_util import create_model
from dependencies.midas.dpt_depth import DPTDepthModel
from dependencies.midas.midas_net import MidasNet
from dependencies.midas.midas_net_custom import MidasNet_small
from dependencies.midas.transforms import NormalizeImage, Resize, PrepareForNet

# from midas.dpt_depth import DPTDepthModel
# from midas.midas_net import MidasNet
# from midas.midas_net_custom import MidasNet_small
# from midas.transforms import Resize, NormalizeImage, PrepareForNet

from config import (
    clip_model_settings,
    torch_model_settings,
    diffusion_model_settings,
    midas_depth_model_settings,
)
from constants import DiffusionModelEnum, MidasModelTypeEnum
from diffusion.models import SecondaryDiffusionImageNet2


def load_clip_model() -> List[clip.model.CLIP]:
    """
    Return List of Clip Models
    """
    logger.info("Load Clip Model")
    clip_models = []
    device = torch_model_settings.device
    if clip_model_settings.ViTB32 is True:
        logger.info("Load ViTB32")
        clip_models.append(
            clip.load("ViT-B/32", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.ViTB16 is True:
        logger.info("Load ViTB16")
        clip_models.append(
            clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.ViTL14 is True:
        logger.info("Load ViTL14")
        clip_models.append(
            clip.load("ViT-L/14", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.RN50 is True:
        logger.info("Load RN50")
        clip_models.append(clip.load("RN50", jit=False)[0].eval().requires_grad_(False).to(device))
    if clip_model_settings.RN50x4 is True:
        logger.info("Load RN50x4")
        clip_models.append(
            clip.load("RN50x4", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.RN50x16 is True:
        logger.info("Load RN50x16")
        clip_models.append(
            clip.load("RN50x16", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.RN50x64 is True:
        logger.info("Load RN50x64")
        clip_models.append(
            clip.load("RN50x64", jit=False)[0].eval().requires_grad_(False).to(device)
        )
    if clip_model_settings.RN101 is True:
        logger.info("Load RN101")
        clip_models.append(clip.load("RN101", jit=False)[0].eval().requires_grad_(False).to(device))
    return clip_models


def load_diffusion_model() -> torch.nn.Module:
    """
    Return Diffusion Model
    """
    logger.info("Load Diffusion Model")
    diffusion_model = diffusion_model_settings.diffusion_model.value
    if diffusion_model == DiffusionModelEnum.DIFFUSION_UNCOND_FINTETUNE_008100_512_BY_512.value:
        model = create_model(
            image_size=512,
            num_channels=256,
            num_res_blocks=2,
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=True,
            attention_resolutions="32, 16, 8",
            num_head_channels=64,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=torch_model_settings.use_fp16,
        )
    elif diffusion_model == DiffusionModelEnum.DIFFUSION_UNCOND_256_BY_256.value:
        model = create_model(
            image_size=256,
            num_channels=256,
            num_res_blocks=2,
            learn_sigma=True,
            class_cond=False,
            use_checkpoint=True,
            attention_resolutions="32, 16, 8",
            num_head_channels=64,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=torch_model_settings.use_fp16,
        )
    else:
        logger.error(f"{diffusion_model} is not supported")
        raise ValueError(f"{diffusion_model} is not supported")
    logger.info(f"Load {diffusion_model} Model")
    model.load_state_dict(torch.load(f"./pytorch_models/{diffusion_model}.pt", map_location="cpu"))
    model.requires_grad_(False).eval().to(torch_model_settings.device)
    return model


def load_secondary_diffusion_model() -> torch.nn.Module:
    """
    Return Secondary Diffusion Model
    """
    logger.info("Load Secondary Diffusion Model")
    if diffusion_model_settings.use_secondary_model:
        secondary_model = SecondaryDiffusionImageNet2()
        secondary_model.load_state_dict(
            torch.load(f"./pytorch_models/secondary_model_imagenet_2.pth", map_location="cpu")
        )
        secondary_model.eval().requires_grad_(False).to(torch_model_settings.device)
        return secondary_model
    else:
        return None


# def init_midas_depth_model(midas_model_type="dpt_large", optimize=True):
def load_midas_depth_model() -> Dict:
    """
    Return Midas Depth Model
    """
    midas_model = None
    net_w = None
    net_h = None
    resize_mode = None
    normalization = None
    midas_model_type = midas_depth_model_settings.midas_model_type.value
    midas_model_path = f"./pytorch_models/{midas_model_type}.pt"
    logger.info(f"Load MiDaS {midas_model_type} Depth Model")
    if midas_model_type == MidasModelTypeEnum.DPT_LARGE.value:  # DPT-Large
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == MidasModelTypeEnum.DPT_HYBRID.value:
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == MidasModelTypeEnum.DPT_HYBRID_NYU.value:
        midas_model = DPTDepthModel(
            path=midas_model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == MidasModelTypeEnum.MIDAS_V21.value:
        midas_model = MidasNet(midas_model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode = "upper_bound"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif midas_model_type == MidasModelTypeEnum.MIDAS_V21_SMALL.value:
        midas_model = MidasNet_small(
            midas_model_path,
            features=64,
            backbone="efficientnet_lite3",
            exportable=True,
            non_negative=True,
            blocks={"expand": True},
        )
        net_w, net_h = 256, 256
        resize_mode = "upper_bound"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        logger.error(f"{midas_model_type} is not supported")
        raise ValueError(f"{midas_model_type} is not supported")
    midas_transform = T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    if torch_model_settings.use_fp16 and torch_model_settings.device == torch.device("cuda"):
        midas_model = midas_model.to(memory_format=torch.channels_last)
        midas_model = midas_model.half()

    midas_model.to(torch_model_settings.device)
    return {
        "mida_model": midas_model,
        "midas_transform": midas_transform,
        "net_w": net_w,
        "net_h": net_h,
        "resize_mode": resize_mode,
        "normalization": normalization,
    }
