from typing import List

import torch
import clip
from loguru import logger

from dependencies.guided_diffusion.guided_diffusion.script_util import create_model

from config import clip_model_settings, torch_model_settings, diffusion_model_settings
from constants import DiffusionModelEnum


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


def load_diffusion_model() -> List:
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
