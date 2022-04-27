from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Settings
    """

    app_name: str = "Disco Diffusion Fast API"
    app_version: str = "0.1.0-dev"
    api_prefix: str = "/api"


settings = Settings()
