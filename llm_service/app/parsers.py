"""Резервный парсер текстового описания клиента без вызова LLM."""

from __future__ import annotations

import json
import re
from typing import Any

DEFAULT_CASE = {
    "age": 37,
    "workclass": "Private",
    "fnlwgt": 100000,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}

EDUCATION_MAP = {
    "preschool": ("Preschool", 1),
    "1st-4th": ("1st-4th", 2),
    "5th-6th": ("5th-6th", 3),
    "7th-8th": ("7th-8th", 4),
    "9th": ("9th", 5),
    "10th": ("10th", 6),
    "11th": ("11th", 7),
    "12th": ("12th", 8),
    "hs-grad": ("HS-grad", 9),
    "some-college": ("Some-college", 10),
    "assoc-voc": ("Assoc-voc", 11),
    "assoc-acdm": ("Assoc-acdm", 12),
    "bachelors": ("Bachelors", 13),
    "masters": ("Masters", 14),
    "prof-school": ("Prof-school", 15),
    "doctorate": ("Doctorate", 16),
}



def _search_int(pattern: str, text: str, default: int) -> int:
    """Ищет целое число по регулярному выражению."""
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else default



def heuristic_parse_case(text: str) -> dict[str, Any]:
    """Разбирает текст клиента в структуру Adult-совместимых признаков.

    Args:
        text: Свободное текстовое описание клиента.

    Returns:
        Словарь признаков в формате, совместимом с пайплайном модели.
    """
    lowered = text.lower()
    result = dict(DEFAULT_CASE)

    result["age"] = _search_int(r"(\d{2})\s*year", text, result["age"])
    result["capital_gain"] = _search_int(r"capital gain[^\d]*(\d+)", text, result["capital_gain"])
    result["capital_loss"] = _search_int(r"capital loss[^\d]*(\d+)", text, result["capital_loss"])
    result["hours_per_week"] = _search_int(r"(\d+)\s*hours? per week", text, result["hours_per_week"])

    for key, (education, education_num) in EDUCATION_MAP.items():
        if key in lowered:
            result["education"] = education
            result["education_num"] = education_num
            break

    for workclass in [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ]:
        if workclass.lower() in lowered:
            result["workclass"] = workclass
            break

    for occupation in [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces"
    ]:
        if occupation.lower() in lowered:
            result["occupation"] = occupation
            break

    if "married" in lowered:
        result["marital_status"] = "Married-civ-spouse"
        result["relationship"] = "Husband" if "male" in lowered else "Wife"
    elif "divorced" in lowered:
        result["marital_status"] = "Divorced"
        result["relationship"] = "Unmarried"
    elif "widowed" in lowered:
        result["marital_status"] = "Widowed"
        result["relationship"] = "Unmarried"

    if "female" in lowered:
        result["sex"] = "Female"
    elif "male" in lowered:
        result["sex"] = "Male"

    for country in [
        "United-States", "Canada", "Germany", "England", "India", "Mexico",
        "Japan", "China", "Philippines", "France", "Italy", "Poland"
    ]:
        if country.lower() in lowered:
            result["native_country"] = country
            break

    for race in ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]:
        if race.lower() in lowered:
            result["race"] = race
            break

    return result



def safe_json_loads(raw_text: str) -> dict[str, Any]:
    """Пытается безопасно распарсить JSON из ответа модели.

    Args:
        raw_text: Строка, предположительно содержащая JSON.

    Returns:
        Словарь признаков.

    Raises:
        ValueError: Если JSON не удалось разобрать.
    """
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if not match:
            raise ValueError("Could not find JSON object in model output.")
        return json.loads(match.group(0))
