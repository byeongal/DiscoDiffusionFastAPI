from typing import Union

import torch
from pydantic import BaseSettings
from constants import DiffusionModelEnum, DiffusionSamplingModeEnum, MidasModelTypeEnum


class AppSettings(BaseSettings):
    """
    Settings about App
    """

    app_name: str = "Disco Diffusion Fast API"
    app_version: str = "0.1.0-dev"
    api_prefix: str = "/api"
    is_debug: bool = True


class TorchModelSettings(BaseSettings):
    """
    Settings about Torch Model
    """

    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True


class DiffusionModelSettings(BaseSettings):
    """
    Settings about Diffusion Model
    """

    diffusion_model: DiffusionModelEnum = (
        DiffusionModelEnum.DIFFUSION_UNCOND_FINTETUNE_008100_512_BY_512
    )
    use_secondary_model: bool = True
    diffusion_sampling_mode: DiffusionSamplingModeEnum = DiffusionSamplingModeEnum.DDIM


class ClipModelSettings(BaseSettings):
    """
    Settings about Clip Model
    """

    ViTB32: bool = True
    ViTB16: bool = True
    ViTL14: bool = False
    RN101: bool = False
    RN50: bool = True
    RN50x4: bool = False
    RN50x16: bool = False
    RN50x64: bool = False


class MidasDepthModelSettings(BaseSettings):
    """
    Settings about Midas Depth Model
    """

    midas_model_type: MidasModelTypeEnum = MidasModelTypeEnum.DPT_LARGE


app_settings = AppSettings()
torch_model_settings = TorchModelSettings()
diffusion_model_settings = DiffusionModelSettings()
clip_model_settings = ClipModelSettings()
midas_depth_model_settings = MidasDepthModelSettings()
