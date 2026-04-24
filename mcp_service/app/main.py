"""FastAPI-обёртка над LLM-сервисом и MCP-клиентом."""

from __future__ import annotations

from fastapi import FastAPI

from app.llm_client import parse_with_llm
from app.mcp_client import run_all_tools
from app.schemas import AnalyzeRequest, AnalyzeResponse, ToolEnvelope

app = FastAPI(title="MCP Credit Underwriting Service", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    """Возвращает статус MCP-сервиса.

    Returns:
        Словарь состояния.
    """
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Выполняет полный анализ текстового кредитного кейса.

    Шаги пайплайна:
    1. Отправка текста в LLM-сервис.
    2. Получение структурированного кейса.
    3. Вызов MCP-тулов через FastMCP Client.
    4. Сбор итогового решения и краткой сводки.

    Args:
        payload: Запрос с текстом клиента.

    Returns:
        Полный результат анализа.
    """
    parsed = await parse_with_llm(payload.text)
    case_payload = parsed["structured_case"]
    tool_results_raw = await run_all_tools(case_payload)
    tool_results = [ToolEnvelope(**item) for item in tool_results_raw]

    approval_probability = next(
        item.result["approval_probability"]
        for item in tool_results
        if item.name == "predict_approval"
    )
    risk_label = next(
        item.result["risk_label"]
        for item in tool_results
        if item.name == "assess_risk"
    )
    score = next(
        item.result["score"]
        for item in tool_results
        if item.name == "calculate_credit_score"
    )

    final_decision = "approve" if approval_probability >= 0.45 and risk_label == "low_risk" else "manual_review"
    summary = (
        f"Score={score}, approval_probability={approval_probability}, "
        f"risk={risk_label}, final_decision={final_decision}."
    )

    return AnalyzeResponse(
        parser_used=parsed["parser_used"],
        structured_case=case_payload,
        tool_results=tool_results,
        final_decision=final_decision,
        summary=summary,
    )
