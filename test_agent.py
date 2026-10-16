"""Pruebas unitarias de la estructura del agente, herramientas y persistencia.

Se ejecutan localmente sin necesidad de realizar llamadas reales al LLM.
Uso:
    python test_agent.py
"""
from __future__ import annotations

import json
import unittest
from langgraph.graph.message import MessagesState
from langchain_core.messages import AIMessage, HumanMessage

from agent import AgentState, build_agent_graph, get_sync_checkpointer
from tools import buscar_pedidos, obtener_detalle_pedido, TOOLS


class TestAgenteReAct(unittest.TestCase):
    """Verifica que la arquitectura cumpla estrictamente con la rúbrica del Módulo 5."""

    def test_01_herencia_messages_state(self) -> None:
        """El estado debe heredar de MessagesState para acumular mensajes vía reducers."""
        self.assertTrue(
            issubclass(AgentState, MessagesState),
            "AgentState debe heredar formalmente de MessagesState.",
        )

    def test_02_herramienta_buscar_pedidos_existente(self) -> None:
        """La herramienta buscar_pedidos debe retornar JSON con los pedidos del cliente 102."""
        res_raw = buscar_pedidos.invoke({"cliente_id": 102})
        data = json.loads(res_raw)
        self.assertEqual(data["cliente_id"], 102)
        self.assertEqual(data["cantidad_pedidos"], 3)
        self.assertEqual(data["monto_total"], 14500.0)
        self.assertIn("P-9277", [p["pedido_id"] for p in data["pedidos"]])

    def test_03_herramienta_buscar_pedidos_no_encontrado(self) -> None:
        """Cliente inexistente debe retornar error controlado sin crashear."""
        res_raw = buscar_pedidos.invoke({"cliente_id": 99999})
        data = json.loads(res_raw)
        self.assertIn("error", data)
        self.assertEqual(data["pedidos"], [])

    def test_04_herramienta_obtener_detalle_pedido(self) -> None:
        """obtener_detalle_pedido debe devolver los items específicos y la dirección."""
        res_raw = obtener_detalle_pedido.invoke({"pedido_id": "P-9277"})
        data = json.loads(res_raw)
        self.assertEqual(data["pedido_id"], "P-9277")
        self.assertEqual(data["estado"], "Entregado")
        self.assertTrue(len(data["items"]) >= 1)
        self.assertIn("Datacenter", data["direccion_entrega"])

    def test_05_validacion_pydantic_cliente_id_invalido(self) -> None:
        """cliente_id <= 0 debe ser rechazado por la validación de Pydantic."""
        with self.assertRaises(Exception):
            buscar_pedidos.invoke({"cliente_id": -5})

    def test_06_topologia_del_grafo(self) -> None:
        """El StateGraph compilado debe contener los nodos 'model' y 'tools'."""
        checkpointer = get_sync_checkpointer(":memory:")
        app = build_agent_graph(checkpointer=checkpointer)

        # Inspección de los nodos registrados en el grafo
        node_names = list(app.get_graph().nodes.keys())
        self.assertIn("model", node_names, "Debe existir el nodo 'model'.")
        self.assertIn("tools", node_names, "Debe existir el nodo 'tools'.")

    def test_07_herramientas_registradas(self) -> None:
        """El agente debe disponer de al menos una herramienta decorada con @tool."""
        self.assertTrue(len(TOOLS) >= 1)
        tool_names = [t.name for t in TOOLS]
        self.assertIn("buscar_pedidos", tool_names)
        self.assertIn("obtener_detalle_pedido", tool_names)


if __name__ == "__main__":
    unittest.main()
