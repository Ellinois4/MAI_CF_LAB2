# Лабораторная работа 2: NLP 

## Идея проекта

Разработан proof-of-concept прототип для банка, который принимает **свободное текстовое описание клиента**, преобразует его в структурированный кейс с помощью LLM, а затем вызывает набор MCP-тулов для расчёта кредитного скора, оценки риска и рекомендации по одобрению заявки.

В качестве учебного датасета используется **Adult / Census Income**. В этой лабораторной работе целевая переменная `income` трактуется как **прокси-сигнал платёжеспособности**: высокий доход не равен кредитоспособности, но подходит для демонстрации пайплайна LLM + MCP + классический ML.

## Что реализовано

- LLM-сервис на **FastAPI** + **Ollama** + `qwen2.5:0.5b`
- MCP-сервис на **FastAPI** + **FastMCP**
- MCP-сервер минимум с двумя тулами
- MCP-клиент, который вызывает тулы сервера
- Набор дополнительных ML-тулов для улучшения качества решения
- Скрипты подготовки данных, обучения моделей и оценки пайплайна
- Исследовательский отчёт по метрикам `accuracy`, `precision`, `recall`, `f1`
- Полная docstring-документация функций

## Пошаговый запуск

### 1. Установить Docker и Docker Compose

Нужен Docker Desktop или docker engine + compose plugin.

### 2. Распаковать архив проекта

```bash
unzip credit_lab_project.zip
cd credit_lab_project
```

### 3. При желании создать `.env`

```bash
cp .env.example .env
```

### 4. Подготовить датасет и обучить модели

Локально, вне Docker:

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r scripts/requirements.txt
python scripts/prepare_data.py
python scripts/train_models.py
python scripts/evaluate_pipeline.py
```

После этого в `artifacts/models/` появятся обученные модели, а в `artifacts/reports/` — метрики и отчёт.

### 5. Поднять сервисы

```bash
docker compose up --build
```

Поднимутся контейнеры:
- `ollama`
- `ollama-init`
- `llm-service`
- `mcp-service`

### 6. Проверить LLM-сервис

```bash
curl -X POST http://localhost:8000/parse_case   -H "Content-Type: application/json"   -d '{
    "text": "Client is a 45 year old married male with Bachelors degree, works as Exec-managerial in private sector, 50 hours per week, capital gain 5000, from United-States"
  }'
```

### 7. Проверить MCP-сервис

```bash
curl -X POST http://localhost:8001/analyze   -H "Content-Type: application/json"   -d '{
    "text": "Client is a 45 year old married male with Bachelors degree, works as Exec-managerial in private sector, 50 hours per week, capital gain 5000, from United-States"
  }'
```

### 8. Запустить smoke-test

```bash
python scripts/smoke_test.py
```

## Формат входного текста

Сервис ожидает **свободное текстовое описание** клиента. Например:

```text
A 37-year-old female from United-States works in private sector as Prof-specialty.
She has Masters degree, is married, works 45 hours per week,
capital gain is 0, capital loss is 0.
```

## MCP-тулы

Обязательные тулы:

1. `calculate_credit_score` — расчёт кредитного скора по структурированному кейсу.
2. `assess_risk` — оценка риска по кейсу.

Дополнительные ML-тулы:

3. `predict_approval` — прогноз вероятности одобрения по модели логистической регрессии.
4. `explain_case` — короткое объяснение факторов решения.

## Основные гипотезы

1. **LLM способен переводить свободный текст клиента в структурированные признаки**, пригодные для скоринга.
2. **Без классического ML качество решения будет низким**, потому что маленькая LLM хуже справляется с точным классификационным решением на табличных данных.
3. **Подключение MCP-тулов с классическим ML заметно повышает метрики** по сравнению с эвристическим LLM-only подходом.
4. Лучший результат даст не один скор, а **комбинация нескольких инструментов**: скоринг, вероятность одобрения, риск-сегмент и текстовое объяснение.
