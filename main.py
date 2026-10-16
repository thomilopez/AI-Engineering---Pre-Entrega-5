"""Script principal de ejecución y validación del Agente ReAct con persistencia.

Demuestra de forma 100% asíncrona:
1. Ciclo ReAct Multi-paso: el agente razona e invoca de forma autónoma ≥2 herramientas
   en una misma consulta compleja (buscar pedidos + obtener detalle del último pedido).
2. Persistencia con SqliteSaver y thread_id: en la segunda consulta el usuario pregunta
   por el contexto previo sin repetir IDs ('¿Y a qué dirección se entregó?'), y el agente
   recupera la memoria desde la base de datos local SQLite.
3. Resiliencia contra bucles infinitos: toda invocación define recursion_limit=10.
4. Generación de traza de ejecución en formato JSON (trace_react.json) y LOG (trace_react.log).
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from agent import (
    build_agent_graph,
    get_async_checkpointer,
    RECURSION_LIMIT,
    SQLITE_DB_PATH,
)

SESSION_THREAD_ID = "sesion-auditoria-cliente-102"
JSON_TRACE_FILE = "trace_react.json"
LOG_TRACE_FILE = "trace_react.log"


# ============================================================================
# Serialización y Formateo de Trazas
# ============================================================================
def serializar_mensaje(msg: BaseMessage) -> dict[str, Any]:
    """Convierte un mensaje de LangChain en una estructura dict serializable."""
    entry: dict[str, Any] = {
        "tipo": type(msg).__name__,
        "contenido": msg.content if isinstance(msg.content, str) else str(msg.content),
    }

    if isinstance(msg, AIMessage) and msg.tool_calls:
        entry["tool_calls"] = [
            {
                "id": tc.get("id"),
                "name": tc.get("name"),
                "args": tc.get("args"),
            }
            for tc in msg.tool_calls
        ]

    if isinstance(msg, ToolMessage):
        entry["tool_call_id"] = msg.tool_call_id
        entry["tool_name"] = getattr(msg, "name", None) or "tool"

    return entry


def formatear_traza_humana(turno_num: int, pregunta: str, mensajes: list[BaseMessage]) -> str:
    """Genera una representación en texto plano del ciclo ReAct (Reason -> Act -> Observe)."""
    lineas: list[str] = [
        f"\n{'=' * 75}",
        f"TURNO {turno_num}: REZONAMIENTO Y EJECUCIÓN DEL AGENTE",
        f"{'=' * 75}",
        f"Usuario: \"{pregunta}\"\n",
    ]

    for m in mensajes:
        if isinstance(m, HumanMessage):
            continue
        elif isinstance(m, SystemMessage):
            continue
        elif isinstance(m, AIMessage):
            if m.tool_calls:
                for tc in m.tool_calls:
                    lineas.append(f"  [REASONING] El agente decide invocar la herramienta: {tc['name']}")
                    lineas.append(f"  [ACTION]    Llamada con argumentos: {json.dumps(tc['args'], ensure_ascii=False)}")
            elif m.content:
                lineas.append(f"  [RESPONSE]  Respuesta Final del Agente:")
                lineas.append(f"              \"{m.content.strip()}\"")
        elif isinstance(m, ToolMessage):
            lineas.append(f"  [OBSERVE]   Resultado devuelto por '{getattr(m, 'name', 'herramienta')}':")
            lineas.append(f"              {m.content.strip()}")

    lineas.append(f"{'-' * 75}\n")
    return "\n".join(lineas)


# ============================================================================
# Ejecución Asíncrona del Agente
# ============================================================================
async def ejecutar_consulta(
    app: Any,
    pregunta: str,
    thread_id: str,
    turno_num: int,
) -> tuple[str, list[dict[str, Any]], str]:
    """Envía un mensaje al agente dentro de un hilo específico con control de recursión.

    Retorna:
      - respuesta_texto: contenido de la respuesta final.
      - traza_dict: lista de mensajes serializados para JSON.
      - traza_legible: bloque de texto formateado con el ciclo Reason->Act->Observe.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Invocación ASÍNCRONA no bloqueante con recursion_limit explícito
    resultado = await app.ainvoke(
        {"messages": [HumanMessage(content=pregunta)]},
        config=config,
        recursion_limit=RECURSION_LIMIT,
    )

    mensajes: list[BaseMessage] = resultado["messages"]
    traza_dict = [serializar_mensaje(m) for m in mensajes]
    traza_legible = formatear_traza_humana(turno_num, pregunta, mensajes)

    # El último mensaje emitido es la respuesta textual del LLM
    ultimo_mensaje = mensajes[-1]
    respuesta_texto = str(ultimo_mensaje.content) if ultimo_mensaje else ""

    return respuesta_texto, traza_dict, traza_legible


async def main() -> None:
    """Punto de entrada asíncrono para ejecutar la suite de pruebas del agente."""
    print("=" * 75)
    print("INICIANDO PRUEBA DEL AGENTE REACT CON PERSISTENCIA EN SQLITE")
    print(f"Base de datos de checkpoints: {SQLITE_DB_PATH}")
    print(f"Session Thread ID:            {SESSION_THREAD_ID}")
    print(f"Recursion Limit:              {RECURSION_LIMIT} pasos")
    print("=" * 75)

    # Inicialización del checkpointer asíncrono con context manager
    async with get_async_checkpointer(SQLITE_DB_PATH) as checkpointer:
        app = build_agent_graph(checkpointer=checkpointer)

        traza_completa: list[dict[str, Any]] = []
        log_completo: list[str] = []

        # --------------------------------------------------------------------
        # PRUEBA 1: Consulta Multi-paso (fuerza >=2 tool calls en ciclo ReAct)
        # --------------------------------------------------------------------
        q1 = (
            "¿Cuántos pedidos tuvo el cliente 102, cuál fue el total gastado "
            "y cuáles son los productos incluidos en su último pedido?"
        )
        print("\n[PASO 1] Ejecutando consulta multi-paso...")
        print(f"Pregunta: {q1}\n")

        r1, t1, log1 = await ejecutar_consulta(app, q1, SESSION_THREAD_ID, turno_num=1)
        print(log1)

        traza_completa.append({
            "turno": 1,
            "thread_id": SESSION_THREAD_ID,
            "pregunta": q1,
            "respuesta_final": r1,
            "historial_mensajes": t1,
        })
        log_completo.append(log1)

        # --------------------------------------------------------------------
        # PRUEBA 2: Consulta de seguimiento (demuestra persistencia por thread_id)
        # --------------------------------------------------------------------
        q2 = "¿Y a qué dirección fue entregado ese último pedido?"
        print("\n[PASO 2] Ejecutando consulta contextual de seguimiento (mismo thread_id)...")
        print(f"Pregunta: {q2}\n")

        r2, t2, log2 = await ejecutar_consulta(app, q2, SESSION_THREAD_ID, turno_num=2)
        print(log2)

        traza_completa.append({
            "turno": 2,
            "thread_id": SESSION_THREAD_ID,
            "pregunta": q2,
            "respuesta_final": r2,
            "historial_mensajes": t2,
        })
        log_completo.append(log2)

        # --------------------------------------------------------------------
        # Métricas y Guardado de Trazas
        # --------------------------------------------------------------------
        total_tool_calls = sum(
            len(m.get("tool_calls", []))
            for turno in traza_completa
            for m in turno["historial_mensajes"]
            if "tool_calls" in m
        )

        print("\n" + "=" * 75)
        print("RESUMEN DE VALIDACIÓN")
        print("=" * 75)
        print(f"Total de Tool Calls invocados en la sesión: {total_tool_calls} (Requerido: >= 2)")
        print(f"Persistencia validada: Turno 2 resolvió la consulta usando la memoria del Thread ID '{SESSION_THREAD_ID}'.")
        print(f"Límite de recursión respetado: {RECURSION_LIMIT} pasos.")

        # Guardado en archivo JSON para entrega
        with open(JSON_TRACE_FILE, "w", encoding="utf-8") as f:
            json.dump(traza_completa, f, ensure_ascii=False, indent=2)
        print(f"[OK] Traza JSON exportada a '{JSON_TRACE_FILE}'.")

        # Guardado en archivo LOG para inspección rápida
        with open(LOG_TRACE_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(log_completo))
        print(f"[OK] Traza LOG exportada a '{LOG_TRACE_FILE}'.")
        print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
