"""Небольшой smoke-test для локальной проверки готового сервиса."""

from __future__ import annotations

import json
import requests


SAMPLE_TEXT = (
    "Client is a 45 year old Male from United-States. "
    "Works in Private as Exec-managerial. "
    "Education is Bachelors with education_num 13. "
    "Marital status is Married-civ-spouse; relationship is Husband. "
    "Race is White. Capital gain is 5000, capital loss is 0. "
    "Works 50 hours per week. Sampling weight is 120000."
)



def main() -> None:
    """Отправляет тестовый кейс в MCP-сервис и печатает ответ."""
    response = requests.post(
        "http://localhost:8001/analyze",
        json={"text": SAMPLE_TEXT},
        timeout=120,
    )
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
