"""Клиент для LLM-сервиса."""

from __future__ import annotations

import httpx

from app.config import settings


async def parse_with_llm(text: str) -> dict:
    """Отправляет текст в LLM-сервис и получает структурированный кейс.

    Args:
        text: Описание клиента.

    Returns:
        Ответ LLM-сервиса в JSON-виде.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.llm_service_url}/parse_case",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()
