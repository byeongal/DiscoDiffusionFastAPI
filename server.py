from fastapi import FastAPI

from api.router import api_router
from config import settings


def get_app() -> FastAPI:
    """
    Return an Initialized FastAPI App.
    """
    fast_api_app = FastAPI(
        title=settings.app_name, version=settings.app_version, debug=settings.is_debug
    )
    fast_api_app.include_router(api_router, prefix=settings.api_prefix)
    return fast_api_app


app = get_app()
