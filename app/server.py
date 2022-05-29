import os
from fastapi import FastAPI

from api.router import api_router
from config import app_settings
from event_handlers import start_app_handler, stop_app_handler


def get_app() -> FastAPI:
    """
    Return an Initialized FastAPI App.
    """
    fast_api_app = FastAPI(
        title=app_settings.app_name, version=app_settings.app_version, debug=app_settings.is_debug
    )
    fast_api_app.include_router(api_router, prefix=app_settings.api_prefix)
    fast_api_app.add_event_handler("startup", start_app_handler(fast_api_app))
    fast_api_app.add_event_handler("shutdown", stop_app_handler(fast_api_app))
    if not os.path.exists("./output_images"):
        os.makedirs("./output_images")
    return fast_api_app


app = get_app()
