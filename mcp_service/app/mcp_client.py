"""Программный клиент FastMCP для вызова тулов сервера."""

from __future__ import annotations

from fastmcp import Client

from app.mcp_server import mcp


async def run_all_tools(case_payload: dict) -> list[dict]:
    """Вызывает все зарегистрированные MCP-тулы по кейсу клиента.

    Args:
        case_payload: Структурированный кейс в виде словаря.

    Returns:
        Список результатов вызовов тулов.
    """
    tool_names = [
        "calculate_credit_score",
        "assess_risk",
        "predict_approval",
        "explain_case",
    ]
    results: list[dict] = []

    client = Client(mcp)
    async with client:
        for tool_name in tool_names:
            response = await client.call_tool(tool_name, {"case": case_payload})
            results.append({"name": tool_name, "result": response.data})
    return results
