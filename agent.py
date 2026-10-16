"""Agente de razonamiento cíclico con memoria persistente (LangGraph + SQLite).

Arquitectura del Grafo:
    START -> model -> [tools_condition] -> tools -> model ... -> END

    - El nodo 'model' invoca al LLM vinculado a las herramientas con llm.bind_tools().
    - La arista condicional 'tools_condition' inspecciona el último mensaje: si contiene
      tool_calls, enruta al nodo 'tools'; de lo contrario, finaliza hacia END.
    - La arista de retorno 'tools -> model' cierra el ciclo ReAct: el agente observa la
      salida de la herramienta en un ToolMessage y vuelve a razonar.
    - Estado: AgentState hereda de MessagesState, asegurando inmutabilidad y acumulación
      de mensajes mediante reducers (add_messages), sin sobrescribir el historial.
    - Persistencia: SqliteSaver / AsyncSqliteSaver almacena snapshots de estado indexados
      por thread_id, permitiendo reanudar conversaciones y recordar contexto previo.
"""
from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from tools import TOOLS

load_dotenv()

# ============================================================================
# Constantes y Configuración
# ============================================================================
DEFAULT_MODEL = "gpt-4o-mini"
RECURSION_LIMIT = 10
SQLITE_DB_PATH = "checkpoints.db"

SYSTEM_PROMPT = (
    "Eres un asistente técnico de gestión comercial y logística. "
    "Tu objetivo es responder las consultas del usuario utilizando las herramientas disponibles. "
    "Sigue estrictamente el ciclo ReAct: analiza la petición, decide qué herramienta llamar con sus "
    "parámetros correctos, observa el resultado retornado y sólo elabora una respuesta final cuando "
    "dispongas de toda la información requerida. Si una consulta requiere varios pasos (ej. listar pedidos "
    "y luego consultar el detalle del último), invoca las herramientas secuencialmente. "
    "Sé conciso, preciso y transparente con los datos obtenidos."
)


# ============================================================================
# Estado del Grafo (hereda de MessagesState)
# ============================================================================
class AgentState(MessagesState):
    """Estado tipado del agente que hereda de MessagesState.

    MessagesState define el campo:
        messages: Annotated[list[BaseMessage], add_messages]
    El reducer add_messages garantiza que cada actualización devuelta por un nodo
    se anexe al final del historial de mensajes, preservando el contexto completo.
    """
    pass


# ============================================================================
# Fábrica de LLM con Tool Binding
# ============================================================================
def get_model(model_name: str | None = None) -> Any:
    """Instancia el ChatModel y lo vincula con las herramientas disponibles."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No se encontró 'OPENAI_API_KEY' en el entorno. "
            "Copia '.env.example' a '.env' y completa tu API Key de OpenAI."
        )

    model = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
    )
    # Vinculación formal de herramientas mediante llm.bind_tools()
    return llm.bind_tools(TOOLS)


# ============================================================================
# Nodos del Grafo
# ============================================================================
async def call_model(state: AgentState) -> dict[str, list[Any]]:
    """Nodo 'model': razona sobre el historial acumulado y genera una respuesta o tool_call.

    Añade un SystemMessage contextual al inicio si la conversación no lo posee,
    e invoca al LLM de forma no bloqueante con .ainvoke().
    """
    llm = get_model()
    messages = list(state["messages"])

    # Inyección de System Prompt si no existe ya en el historial
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response = await llm.ainvoke(messages)
    # Retorna solo la actualización parcial. LangGraph la fusiona con add_messages.
    return {"messages": [response]}


# ============================================================================
# Construcción del StateGraph
# ============================================================================
def build_agent_graph(checkpointer: Any = None):
    """Construye, conecta y compila el StateGraph del agente ReAct.

    Topología:
      - Nodos: 'model' (razonamiento) y 'tools' (ejecución de herramientas).
      - Aristas fijas:
          START -> model
          tools -> model  (arista de retorno que cierra el ciclo ReAct)
      - Arista condicional:
          model -> tools_condition -> ['tools', END]
    """
    builder = StateGraph(AgentState)

    # 1. Registro de nodos de trabajo
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    # 2. Aristas de control de flujo
    builder.add_edge(START, "model")

    # 3. Arista condicional basada en tools_condition
    builder.add_conditional_edges(
        "model",
        tools_condition,
    )

    # 4. Arista de retorno al modelo para completar el ciclo ReAct (Reason -> Act -> Observe)
    builder.add_edge("tools", "model")

    # 5. Compilación del grafo con el checkpointer provisto
    return builder.compile(checkpointer=checkpointer)


# ============================================================================
# Utilidades de Persistencia (SqliteSaver)
# ============================================================================
def get_async_checkpointer(db_path: str = SQLITE_DB_PATH) -> AsyncSqliteSaver:
    """Inicializa un checkpointer asíncrono respaldado por SQLite."""
    return AsyncSqliteSaver.from_conn_string(db_path)


def get_sync_checkpointer(db_path: str = SQLITE_DB_PATH) -> SqliteSaver:
    """Inicializa un checkpointer síncrono respaldado por SQLite."""
    import sqlite3

    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)
