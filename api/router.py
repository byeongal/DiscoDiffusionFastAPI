from fastapi import APIRouter
from api import health, generate

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"], prefix="/health")
api_router.include_router(generate.router, tags=["generate"], prefix="/generate")
