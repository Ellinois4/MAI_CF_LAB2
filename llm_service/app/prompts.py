"""Промпты для LLM-сервиса."""


def build_case_extraction_prompt(text: str) -> str:
    """Собирает промпт для извлечения признаков клиента из текста.

    Args:
        text: Входное текстовое описание клиента.

    Returns:
        Готовый промпт для модели Ollama/Qwen.
    """
    return f"""
You are an information extraction system for a bank credit pre-scoring prototype.
Extract a client profile from the text below.
Return strictly valid JSON only.
Use these keys exactly:
age, workclass, fnlwgt, education, education_num, marital_status, occupation,
relationship, race, sex, capital_gain, capital_loss, hours_per_week,
native_country.

Rules:
- Use Adult dataset compatible values whenever possible.
- If a value is missing, fill a sensible default.
- marital-status must become marital_status.
- capital-gain must become capital_gain.
- capital-loss must become capital_loss.
- hours-per-week must become hours_per_week.
- native-country must become native_country.
- education-num must become education_num.
- Return one JSON object and nothing else.

Text:
{text}
""".strip()
