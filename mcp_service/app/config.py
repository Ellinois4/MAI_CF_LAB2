"""Конфигурация MCP-сервиса."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Набор конфигурационных параметров сервиса."""

    def __init__(self) -> None:
        """Считывает настройки окружения."""
        self.llm_service_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8000")
        self.model_dir = Path(os.getenv("MODEL_DIR", "/app/models"))


settings = Settings()
