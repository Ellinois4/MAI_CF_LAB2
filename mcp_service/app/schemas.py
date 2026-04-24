"""Pydantic-схемы MCP-сервиса."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ClientCase(BaseModel):
    """Структурированный кейс клиента для скоринга."""

    age: int
    workclass: str
    fnlwgt: int = 100000
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


class AnalyzeRequest(BaseModel):
    """Запрос на полный анализ клиента по тексту."""

    text: str = Field(..., description="Свободное текстовое описание клиента.")


class ToolEnvelope(BaseModel):
    """Обёртка для ответа конкретного MCP-тула."""

    name: str
    result: dict[str, Any]


class AnalyzeResponse(BaseModel):
    """Ответ объединённого пайплайна анализа клиента."""

    parser_used: str
    structured_case: dict[str, Any]
    tool_results: list[ToolEnvelope]
    final_decision: str
    summary: str
