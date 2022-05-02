from enum import Enum


class HealthStatusEnum(Enum):
    """
    Possible values for the `model.health.HealthStatus.status`.
    """

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class DiffusionModelEnum(Enum):
    """
    Possible values for the `diffusion_model`.
    """

    DIFFUSION_UNCOND_256_BY_256 = "256x256_diffusion_uncond"
    DIFFUSION_UNCOND_FINTETUNE_008100_512_BY_512 = "512x512_diffusion_uncond_finetune_008100"
    SECONDARY_MODEL_IMAGENET_2 = "secondary_model_imagenet_2"


class DiffusionSamplingModeEnum(Enum):
    """
    Possible values for the `diffusion_sampling_mode`.
    """

    PLMS = "plms"
    DDIM = "ddim"
