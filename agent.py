"""Agente ReAct cíclico con LangGraph + persistencia SQLite.

Grafo: START -> agent (LLM con bind_tools) -> [tools_condition] -> tools (ToolNode) -> agent ...
El ciclo termina SOLO cuando el LLM deja de pedir herramientas (tools_condition -> END).
Estado: MessagesState (messages con reducer add_messages: se acumulan, no se sobrescriben).
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from tools import TOOLS

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
RECURSION_LIMIT = 10
DB_PATH = "checkpoints.db"


def get_llm():
    """LLM vinculado a las herramientas. Sin OPENAI_API_KEY falla con mensaje claro."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Falta OPENAI_API_KEY en el .env (ver .env.example).")
    return ChatOpenAI(model=MODEL_NAME, temperature=0).bind_tools(TOOLS)


async def call_model(state: MessagesState) -> dict:
    """Nodo 'agent': razona sobre el estado y decide (responder o pedir tool)."""
    llm_with_tools = get_llm()
    response = await llm_with_tools.ainvoke(state["messages"])
    # Devolvemos actualización parcial; add_messages la ANEXA al historial.
    return {"messages": [response]}


def build_graph(checkpointer) -> object:
    """Construye y compila el StateGraph con persistencia."""
    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))  # ejecuta tools y devuelve ToolMessages
    builder.add_edge(START, "agent")
    # Arista CONDICIONAL: si el último mensaje trae tool_calls -> "tools", si no -> END.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")  # <-- el CICLO ReAct: observar y volver a razonar
    return builder.compile(checkpointer=checkpointer)


def new_checkpointer() -> AsyncSqliteSaver:
    """Checkpointer local persistente (sobrevive reinicios: archivo .db)."""
    return AsyncSqliteSaver.from_conn_string(DB_PATH)
