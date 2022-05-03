from typing import Callable

from fastapi import FastAPI
from loguru import logger
from utils import (
    load_clip_model,
    load_lips,
    load_diffusion_model,
    load_secondary_diffusion_model,
    load_midas_depth_model,
    clear_memory,
)


def _startup_model(app: FastAPI) -> None:
    app.state.clip_models = load_clip_model()
    app.state.lpips_model = load_lips()
    app.state.diffusion_model = load_diffusion_model()
    app.state.secondary_diffusion_model = load_secondary_diffusion_model()
    app.state.midas_depth_model = load_midas_depth_model()
    clear_memory()


def _shutdown_model(app: FastAPI) -> None:
    del app.state.clip_models
    del app.state.diffusion_model
    del app.state.secondary_diffusion_model
    del app.state.midas_depth_model


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
