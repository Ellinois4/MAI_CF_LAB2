"""Утилиты инференса классических ML-моделей."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.schemas import ClientCase

MODEL_FILES = {
    "approval": "approval_logreg.joblib",
    "risk": "risk_random_forest.joblib",
}

_cached_models: dict[str, object] = {}



def _case_to_frame(case: ClientCase) -> pd.DataFrame:
    """Преобразует кейс клиента в DataFrame из одной строки.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Однострочный DataFrame для sklearn-пайплайна.
    """
    return pd.DataFrame([case.model_dump()])



def load_model(model_name: str):
    """Загружает модель с диска с простым кэшем.

    Args:
        model_name: Имя модели из словаря MODEL_FILES.

    Returns:
        Загруженный sklearn-объект.
    """
    if model_name not in _cached_models:
        model_path = Path(settings.model_dir) / MODEL_FILES[model_name]
        _cached_models[model_name] = joblib.load(model_path)
    return _cached_models[model_name]



def predict_approval_probability(case: ClientCase) -> float:
    """Оценивает вероятность положительного класса по модели одобрения.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Вероятность класса `income > 50K`.
    """
    model = load_model("approval")
    frame = _case_to_frame(case)
    return float(model.predict_proba(frame)[0, 1])



def predict_risk_label(case: ClientCase) -> int:
    """Оценивает риск-класс клиента.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Метка риска: 0 — низкий риск, 1 — высокий риск.
    """
    model = load_model("risk")
    frame = _case_to_frame(case)
    return int(model.predict(frame)[0])



def build_feature_summary(case: ClientCase) -> list[str]:
    """Строит компактное объяснение факторов, влияющих на решение.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        Список текстовых факторов.
    """
    factors: list[str] = []
    if case.education_num >= 13:
        factors.append("Высокий уровень образования повышает итоговую оценку")
    if case.hours_per_week >= 45:
        factors.append("Большая занятость в неделю интерпретируется как стабильность занятости")
    if case.capital_gain > 0:
        factors.append("Наличие capital gain повышает ожидаемую платёжеспособность")
    if case.age < 25:
        factors.append("Малый возраст уменьшает устойчивость прогноза")
    if case.capital_loss > 1500:
        factors.append("Существенный capital loss повышает риск")
    if not factors:
        factors.append("Сильных выраженных факторов не обнаружено, решение основано на совокупности признаков")
    return factors
