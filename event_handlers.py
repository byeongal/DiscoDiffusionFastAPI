from typing import Callable

from fastapi import FastAPI
from loguru import logger
from utils import load_clip_model


def _startup_model(app: FastAPI) -> None:
    app.state.clip_models = load_clip_model()


def _shutdown_model(app: FastAPI) -> None:
    app.state.clip_models = None


def start_app_handler(app: FastAPI) -> Callable:
    """
    Event Handler to be used when the server starts.
    """

    def startup() -> None:
        logger.info("Running app start handler.")
        _startup_model(app)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:
    """
    Event Handler to be used when the server stops.
    """

    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        _shutdown_model(app)

    return shutdown
