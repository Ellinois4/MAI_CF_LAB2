"""Клиент для обращения к Ollama API."""

from __future__ import annotations

import os
import httpx


class OllamaClient:
    """Минимальный асинхронный клиент для Ollama generate API."""

    def __init__(self) -> None:
        """Инициализирует клиент из переменных окружения."""
        self.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

    async def generate(self, prompt: str) -> str:
        """Генерирует ответ модели через Ollama.

        Args:
            prompt: Промпт для модели.

        Returns:
            Текстовый ответ модели.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
