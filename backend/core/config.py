"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Data
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory containing village bundle sub-folders.",
    )

    # AI keys (optional — modules degrade gracefully when absent)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    hf_token:       str = Field(default="", alias="HF_TOKEN")

    # Pipeline defaults
    search_radius_m:  float = 15.0
    flag_threshold:   float = 0.15

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "populate_by_name": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
