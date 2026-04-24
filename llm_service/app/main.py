"""FastAPI-приложение LLM-сервиса."""

from __future__ import annotations

from fastapi import FastAPI

from app.ollama_client import OllamaClient
from app.parsers import heuristic_parse_case, safe_json_loads
from app.prompts import build_case_extraction_prompt
from app.schemas import ParseRequest, ParseResponse

app = FastAPI(title="LLM Credit Case Parser", version="1.0.0")
ollama_client = OllamaClient()


@app.get("/health")
async def health() -> dict:
    """Возвращает статус LLM-сервиса.

    Returns:
        Служебный словарь статуса.
    """
    return {"status": "ok"}


@app.post("/parse_case", response_model=ParseResponse)
async def parse_case(payload: ParseRequest) -> ParseResponse:
    """Разбирает текст клиента в структурированный кейс.

    Сначала используется Ollama/Qwen. Если модель вернула невалидный JSON
    или недоступна, включается резервный эвристический парсер.

    Args:
        payload: Текстовое описание клиента.

    Returns:
        Структурированный кейс и метаданные о способе парсинга.
    """
    prompt = build_case_extraction_prompt(payload.text)
    try:
        raw_output = await ollama_client.generate(prompt)
        structured = safe_json_loads(raw_output)
        return ParseResponse(
            structured_case=structured,
            parser_used="ollama_qwen",
            raw_model_output=raw_output,
        )
    except Exception:
        structured = heuristic_parse_case(payload.text)
        return ParseResponse(
            structured_case=structured,
            parser_used="heuristic_fallback",
            raw_model_output=None,
        )
