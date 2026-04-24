"""Общие константы и утилиты для учебного проекта."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "artifacts" / "models"
REPORTS_DIR = ROOT / "artifacts" / "reports"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation", "relationship",
    "race", "sex", "native_country"
]

NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]

TARGET_COLUMN = "income"



def ensure_directories() -> None:
    """Создаёт рабочие директории проекта при их отсутствии."""
    for directory in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)



def build_narrative(row: pd.Series) -> str:
    """Преобразует строку табличного кейса в естественное описание.

    Args:
        row: Строка DataFrame с признаками клиента.

    Returns:
        Англоязычное текстовое описание клиента.
    """
    return (
        f"Client is a {int(row['age'])} year old {row['sex']} from {row['native_country']}. "
        f"Works in {row['workclass']} as {row['occupation']}. "
        f"Education is {row['education']} with education_num {int(row['education_num'])}. "
        f"Marital status is {row['marital_status']}; relationship is {row['relationship']}. "
        f"Race is {row['race']}. "
        f"Capital gain is {int(row['capital_gain'])}, capital loss is {int(row['capital_loss'])}. "
        f"Works {int(row['hours_per_week'])} hours per week. "
        f"Sampling weight is {int(row['fnlwgt'])}."
    )



def generate_synthetic_adult(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Генерирует синтетический Adult-подобный датасет для smoke-тестов.

    Args:
        n_rows: Число строк.
        seed: Инициализация генератора случайных чисел.

    Returns:
        DataFrame с колонками Adult-формата.
    """
    rng = np.random.default_rng(seed)
    educations = ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate", "Assoc-voc"]
    education_num_map = {
        "HS-grad": 9,
        "Some-college": 10,
        "Assoc-voc": 11,
        "Bachelors": 13,
        "Masters": 14,
        "Doctorate": 16,
    }
    occupations = ["Adm-clerical", "Sales", "Craft-repair", "Prof-specialty", "Exec-managerial", "Other-service"]
    workclasses = ["Private", "State-gov", "Local-gov", "Self-emp-not-inc"]
    marital_statuses = ["Never-married", "Married-civ-spouse", "Divorced"]
    relationships = ["Not-in-family", "Husband", "Wife", "Unmarried"]
    races = ["White", "Black", "Asian-Pac-Islander", "Other"]
    sexes = ["Male", "Female"]
    countries = ["United-States", "Canada", "Germany", "India", "Mexico", "England"]

    rows = []
    for _ in range(n_rows):
        education = rng.choice(educations)
        occupation = rng.choice(occupations)
        workclass = rng.choice(workclasses)
        marital_status = rng.choice(marital_statuses)
        relationship = rng.choice(relationships)
        sex = rng.choice(sexes)
        age = int(rng.integers(18, 70))
        hours = int(rng.integers(20, 65))
        capital_gain = int(rng.choice([0, 0, 0, 0, 1500, 5000, 10000]))
        capital_loss = int(rng.choice([0, 0, 0, 0, 500, 1000, 2000]))
        fnlwgt = int(rng.integers(20000, 300000))
        education_num = education_num_map[education]
        race = rng.choice(races)
        country = rng.choice(countries)

        score = 0.15
        score += 0.18 if education_num >= 13 else 0.0
        score += 0.16 if occupation in {"Exec-managerial", "Prof-specialty"} else 0.0
        score += 0.08 if marital_status == "Married-civ-spouse" else 0.0
        score += 0.12 if hours >= 45 else 0.0
        score += 0.14 if capital_gain > 0 else 0.0
        score -= 0.08 if age < 25 else 0.0
        score -= 0.06 if occupation == "Other-service" else 0.0
        prob = min(max(score, 0.02), 0.95)
        income = ">50K" if rng.random() < prob else "<=50K"

        rows.append({
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education_num": education_num,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "hours_per_week": hours,
            "native_country": country,
            "income": income,
        })
    return pd.DataFrame(rows)
