from fastapi import APIRouter
from api import health

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"], prefix="/health")
