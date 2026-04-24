"""Определение FastMCP-сервера и его тулов."""

from __future__ import annotations

from fastmcp import FastMCP

from app.ml_tools import (
    assess_risk_logic,
    calculate_credit_score_logic,
    explain_case_logic,
    predict_approval_logic,
)
from app.schemas import ClientCase

mcp = FastMCP("credit-underwriting-mcp")


@mcp.tool

def calculate_credit_score(case: ClientCase) -> dict:
    """Рассчитывает кредитный скор клиента.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Словарь со скором и вероятностью положительного решения.
    """
    return calculate_credit_score_logic(case)


@mcp.tool

def assess_risk(case: ClientCase) -> dict:
    """Возвращает риск-сегмент клиента.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Словарь с меткой риска.
    """
    return assess_risk_logic(case)


@mcp.tool

def predict_approval(case: ClientCase) -> dict:
    """Возвращает рекомендацию по одобрению.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Словарь с вероятностью и рекомендацией.
    """
    return predict_approval_logic(case)


@mcp.tool

def explain_case(case: ClientCase) -> dict:
    """Объясняет факторы, влияющие на решение по кейсу.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Словарь с кратким объяснением.
    """
    return explain_case_logic(case)
