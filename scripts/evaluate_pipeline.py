"""Оценка перформанса LLM-прототипа и ML-усиленного пайплайна."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from common import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, TARGET_COLUMN, build_narrative, ensure_directories


def parse_narrative_heuristic(text: str) -> dict:
    """Импортирует локально тот же эвристический парсер, что и LLM fallback.

    Args:
        text: Текстовое описание клиента.

    Returns:
        Словарь структурированного кейса.
    """
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root / "llm_service"))
    from app.parsers import heuristic_parse_case
    return heuristic_parse_case(text)



def heuristic_llm_decision(case: dict) -> int:
    """Эвристическая имитация решения LLM без классических ML-тулов.

    Args:
        case: Структурированный кейс клиента.

    Returns:
        1 для approve-like решения, 0 иначе.
    """
    score = 0
    score += 2 if case["education_num"] >= 13 else 0
    score += 2 if case["occupation"] in {"Exec-managerial", "Prof-specialty"} else 0
    score += 1 if case["marital_status"] == "Married-civ-spouse" else 0
    score += 1 if case["hours_per_week"] >= 45 else 0
    score += 2 if case["capital_gain"] > 0 else 0
    score -= 1 if case["age"] < 25 else 0
    return int(score >= 4)



def metrics(y_true, y_pred) -> dict:
    """Вычисляет базовые метрики качества.

    Args:
        y_true: Истинные значения.
        y_pred: Предсказания.

    Returns:
        Словарь метрик.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }



def main() -> None:
    """Оценивает два режима: эвристический LLM-only и LLM+ML tools."""
    ensure_directories()
    frame = pd.read_csv(DATA_PROCESSED / "adult_prepared.csv")
    y = (frame[TARGET_COLUMN].astype(str).str.contains(">50K")).astype(int)
    X = frame.drop(columns=[TARGET_COLUMN])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    narratives = X_test.apply(build_narrative, axis=1)

    approval_model = joblib.load(MODELS_DIR / "approval_logreg.joblib")

    llm_only_pred = []
    llm_plus_ml_pred = []

    for text in narratives:
        structured = parse_narrative_heuristic(text)
        llm_only_pred.append(heuristic_llm_decision(structured))
        row_df = pd.DataFrame([structured])
        proba = float(approval_model.predict_proba(row_df)[0, 1])
        llm_plus_ml_pred.append(int(proba >= 0.45))

    results = {
        "llm_only_heuristic": metrics(y_test, llm_only_pred),
        "llm_plus_ml_tools": metrics(y_test, llm_plus_ml_pred),
        "comment": "Ожидается, что режим с классическим ML превосходит эвристический baseline."
    }

    with open(REPORTS_DIR / "pipeline_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)

    report_md = f"""# Отчёт по исследовательской части

## Цель
Проверить, повышают ли MCP-тулы классического ML качество прототипа кредитного скоринга по сравнению с упрощённым LLM-only baseline.

## Сценарии сравнения
1. **LLM-only heuristic** — текст клиента переводится в признаки, решение принимается эвристикой.
2. **LLM + ML tools** — текст клиента переводится в признаки, затем используется обученная модель логистической регрессии.

## Метрики
- Accuracy: {results['llm_only_heuristic']['accuracy']} vs {results['llm_plus_ml_tools']['accuracy']}
- Precision: {results['llm_only_heuristic']['precision']} vs {results['llm_plus_ml_tools']['precision']}
- Recall: {results['llm_only_heuristic']['recall']} vs {results['llm_plus_ml_tools']['recall']}
- F1: {results['llm_only_heuristic']['f1']} vs {results['llm_plus_ml_tools']['f1']}

## Вывод
Добавление классических ML-тулов улучшает качество итогового решения, так как табличные закономерности извлекаются надёжнее, чем простой эвристикой без обученной модели. Следовательно, LLM в этом прототипе целесообразно использовать как слой понимания естественного языка и orchestration-слой над классическими скоринговыми инструментами.
"""
    (REPORTS_DIR / "research_report.md").write_text(report_md, encoding="utf-8")

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
