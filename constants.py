from enum import Enum


class HealthStatusEnum(Enum):
    """
    Possible values for the `model.health.HealthStatus.status`.
    """

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
