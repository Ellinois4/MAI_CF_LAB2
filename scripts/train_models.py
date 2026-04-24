"""Обучение классических ML-моделей для MCP-тулов."""

from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    CATEGORICAL_COLUMNS,
    DATA_PROCESSED,
    MODELS_DIR,
    NUMERIC_COLUMNS,
    REPORTS_DIR,
    TARGET_COLUMN,
    ensure_directories,
)



def build_preprocessor() -> ColumnTransformer:
    """Создаёт общий препроцессор для числовых и категориальных признаков.

    Returns:
        ColumnTransformer для sklearn-пайплайнов.
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLUMNS),
        ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
    ])



def evaluate_binary(y_true, y_pred) -> dict:
    """Считает основные метрики бинарной классификации.

    Args:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.

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
    """Обучает логистическую регрессию и случайный лес, сохраняет артефакты."""
    ensure_directories()
    frame = pd.read_csv(DATA_PROCESSED / "adult_prepared.csv")
    X = frame.drop(columns=[TARGET_COLUMN]).copy()
    y = (frame[TARGET_COLUMN].astype(str).str.contains(">50K")).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    approval_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    approval_model.fit(X_train, y_train)
    approval_pred = approval_model.predict(X_test)

    risk_target = 1 - y
    _, _, risk_train, risk_test = train_test_split(
        X, risk_target, test_size=0.2, random_state=42, stratify=risk_target
    )
    risk_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    risk_model.fit(X_train, risk_train)
    risk_pred = risk_model.predict(X_test)

    joblib.dump(approval_model, MODELS_DIR / "approval_logreg.joblib")
    joblib.dump(risk_model, MODELS_DIR / "risk_random_forest.joblib")

    report = {
        "approval_model": evaluate_binary(y_test, approval_pred),
        "risk_model": evaluate_binary(risk_test, risk_pred),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    with open(REPORTS_DIR / "training_metrics.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
