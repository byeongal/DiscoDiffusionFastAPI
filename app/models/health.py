from pydantic import BaseModel
from constants import HealthStatusEnum


class HealthStatusResult(BaseModel):
    """
    Health Status Result Model
    """

    status: HealthStatusEnum
