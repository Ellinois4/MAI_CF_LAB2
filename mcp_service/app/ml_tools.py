"""Бизнес-логика ML-тулов для MCP-сервиса."""

from __future__ import annotations

from app.inference import build_feature_summary, predict_approval_probability, predict_risk_label
from app.schemas import ClientCase



def calculate_credit_score_logic(case: ClientCase) -> dict:
    """Рассчитывает кредитный скор по вероятности положительного класса.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Словарь со скором и интерпретацией.
    """
    probability = predict_approval_probability(case)
    score = int(300 + probability * 550)
    band = "high" if score >= 700 else "medium" if score >= 580 else "low"
    return {
        "score": score,
        "probability": round(probability, 4),
        "band": band,
    }



def assess_risk_logic(case: ClientCase) -> dict:
    """Оценивает риск клиента по модели сегментации риска.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Словарь с оценкой риска.
    """
    risk = predict_risk_label(case)
    label = "high_risk" if risk == 1 else "low_risk"
    return {
        "risk_label": label,
        "risk_code": risk,
    }



def predict_approval_logic(case: ClientCase) -> dict:
    """Возвращает рекомендацию по одобрению заявки.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Словарь с вероятностью и бинарной рекомендацией.
    """
    probability = predict_approval_probability(case)
    recommendation = "approve" if probability >= 0.45 else "review_or_decline"
    return {
        "approval_probability": round(probability, 4),
        "recommendation": recommendation,
    }



def explain_case_logic(case: ClientCase) -> dict:
    """Строит короткое текстовое объяснение решения.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Словарь с факторами и текстовой сводкой.
    """
    factors = build_feature_summary(case)
    summary = "; ".join(factors)
    return {
        "top_factors": factors,
        "summary": summary,
    }
