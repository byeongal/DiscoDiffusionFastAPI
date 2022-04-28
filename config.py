import torch
from pydantic import BaseSettings
from constants import DiffusionModelEnum, DiffusionSamplingModeEnum


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

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True


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


app_settings = AppSettings()
torch_model_settings = TorchModelSettings()
clip_model_settings = ClipModelSettings()
