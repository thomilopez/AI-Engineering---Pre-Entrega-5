"""Demo multi-paso del ciclo ReAct + persistencia por thread_id (100% async).

Turno 1: '¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?'
  -> el agente DEBE llamar a buscar_pedidos(cliente_id=102), observar y responder.
Turno 2 (mismo thread_id): '¿Y el último?'
  -> el agente responde usando el checkpoint (recuerda que hablábamos del 102),
     sin que el usuario repita el ID. Si hace falta, re-invoca la tool.

Todo .ainvoke lleva recursion_limit=10 (anti bucles infinitos).
La traza ReAct se guarda en trace_react.json para la entrega.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import HumanMessage
from agent import build_graph, new_checkpointer, RECURSION_LIMIT

THREAD_ID = "demo-cliente-102"


def serializar_mensaje(msg: Any) -> dict[str, Any]:
    """Convierte un mensaje LangChain a dict logueable."""
    data: dict[str, Any] = {"tipo": type(msg).__name__, "contenido": getattr(msg, "content", "")}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = tool_calls
    if hasattr(msg, "tool_call_id"):
        data["tool_call_id"] = msg.tool_call_id
    if hasattr(msg, "name") and msg.name:
        data["tool_name"] = msg.name
    return data


async def preguntar(app, texto: str, thread_id: str) -> tuple[str, list[dict]]:
    """Envía un mensaje al grafo y devuelve (respuesta_final, traza_de_turno)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=texto)]},
        config=config,
        recursion_limit=RECURSION_LIMIT,  # techo anti bucles infinitos
    )
    traza = [serializar_mensaje(m) for m in result["messages"]]
    respuesta = result["messages"][-1].content
    return str(respuesta), traza


async def main() -> None:
    async with new_checkpointer() as checkpointer:
        app = build_graph(checkpointer)
        traza_total: list[dict] = []

        print("=== TURNO 1 (fuerza 1er tool call) ===")
        q1 = "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"
        print(f"Usuario: {q1}")
        r1, t1 = await preguntar(app, q1, THREAD_ID)
        print(f"Agente: {r1}\n")
        traza_total.append({"turno": 1, "pregunta": q1, "respuesta": r1, "mensajes": t1})

        print("=== TURNO 2 (mismo thread_id: prueba memoria) ===")
        q2 = "¿Y el último?"
        print(f"Usuario: {q2}")
        r2, t2 = await preguntar(app, q2, THREAD_ID)
        print(f"Agente: {r2}\n")
        traza_total.append({"turno": 2, "pregunta": q2, "respuesta": r2, "mensajes": t2})

        tool_calls_totales = sum(
            1 for turno in traza_total for m in turno["mensajes"] if "tool_calls" in m
        )
        print(f"Tool calls totales en la sesión: {tool_calls_totales} (esperado >= 2)")

        with open("trace_react.json", "w", encoding="utf-8") as f:
            json.dump(traza_total, f, ensure_ascii=False, indent=2)
        print("Traza guardada en trace_react.json")


if __name__ == "__main__":
    asyncio.run(main())
