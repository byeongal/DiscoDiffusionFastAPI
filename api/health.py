from fastapi import APIRouter
from models.health import HealthStatusResult
from constants import HealthStatusEnum

router = APIRouter()


@router.get("/ping", response_model=HealthStatusResult, name="health_status")
def get_health_status() -> HealthStatusResult:
    """
    Return Server Status
    """
    health_status = HealthStatusResult(status=HealthStatusEnum.HEALTHY.value)
    return health_status
