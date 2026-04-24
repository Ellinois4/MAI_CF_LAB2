"""Pydantic-схемы для LLM-сервиса."""

from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    """Запрос на разбор текстового описания клиента."""

    text: str = Field(..., description="Свободное текстовое описание клиента.")


class ParseResponse(BaseModel):
    """Ответ LLM-сервиса со структурированными признаками клиента."""

    structured_case: dict
    parser_used: str
    raw_model_output: str | None = None
