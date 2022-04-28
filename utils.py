from typing import List

import clip
from loguru import logger

from config import clip_model_settings, torch_model_settings


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
