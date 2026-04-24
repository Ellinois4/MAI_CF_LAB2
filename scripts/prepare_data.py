"""Подготовка датасета Adult из UCI для лабораторной работы."""

from __future__ import annotations

import pandas as pd

from ucimlrepo import fetch_ucirepo

from common import DATA_PROCESSED, ensure_directories


def load_uci_adult() -> pd.DataFrame:
    """Загружает датасет Adult из репозитория UCI через ucimlrepo.

    Returns:
        Подготовленный DataFrame, содержащий признаки и целевую переменную income.
    """
    adult = fetch_ucirepo(id=2)

    features = adult.data.features.copy()
    targets = adult.data.targets.copy()

    if isinstance(targets, pd.DataFrame):
        target_column = targets.columns[0]
        features["income"] = targets[target_column]
    else:
        features["income"] = targets

    return features


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Приводит названия и строковые значения колонок к единому формату проекта.

    Args:
        frame: Исходный DataFrame.

    Returns:
        Нормализованный DataFrame.
    """
    frame = frame.copy()
    frame.columns = [column.strip().replace("-", "_") for column in frame.columns]

    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.strip()
                .str.replace(".", "", regex=False)
            )

    return frame


def main() -> None:
    """Загружает Adult из UCI, нормализует и сохраняет в processed-папку."""
    ensure_directories()

    frame = load_uci_adult()
    frame = normalize_frame(frame)

    output_path = DATA_PROCESSED / "adult_prepared.csv"
    frame.to_csv(output_path, index=False)

    print(f"Saved dataset to {output_path}")
    print("Source used: ucimlrepo (UCI Adult, id=2)")
    print(frame.head(3).to_string())


if __name__ == "__main__":
    main()