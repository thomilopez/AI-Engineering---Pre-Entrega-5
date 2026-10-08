"""Herramientas del agente: simulan una base de datos de pedidos.

El LLM decide usarlas SOLO a partir del docstring + esquema Pydantic.
Nada de rutas manuales if/else: la autonomía nace de descripciones precisas.
"""
from __future__ import annotations

import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# --- Base de datos simulada (en memoria, solo lectura: Least Privilege) ---
_FAKE_DB: dict[int, list[dict]] = {
    102: [
        {"pedido_id": "P-9001", "fecha": "2026-03-02", "total": 4200.0, "items": 2},
        {"pedido_id": "P-9144", "fecha": "2026-05-19", "total": 5300.0, "items": 4},
        {"pedido_id": "P-9277", "fecha": "2026-09-01", "total": 5000.0, "items": 1},
    ],
    205: [
        {"pedido_id": "P-8100", "fecha": "2026-01-11", "total": 1200.0, "items": 1},
    ],
}


class BuscarPedidosInput(BaseModel):
    """Esquema de entrada validado con Pydantic antes de tocar la 'DB'."""

    cliente_id: int = Field(
        ...,
        ge=1,
        le=999999,
        description="ID numérico del cliente, por ejemplo 102. Debe ser un entero positivo.",
    )


@tool(args_schema=BuscarPedidosInput)
def buscar_pedidos(cliente_id: int) -> str:
    """Consulta el historial de pedidos de un cliente en la base de datos de ventas.

    USÁ ESTA HERRAMIENTA cuando el usuario pregunte cuántos pedidos tuvo un cliente,
    cuál fue el monto total acumulado, cuál fue el último pedido, o cualquier resumen
    del historial de compras de un cliente específico (por ejemplo: '¿Cuántos pedidos
    tuvo el cliente 102 y cuál fue el total?').

    Entrada: un único entero `cliente_id` (ejemplo: 102).
    Salida: un JSON string con la lista de pedidos del cliente. Cada pedido contiene
    `pedido_id` (string como 'P-9277'), `fecha` (YYYY-MM-DD), `total` (monto en
    dólares como número) e `items` (cantidad de artículos). Si el cliente no existe,
    devuelve un JSON con `pedidos` vacío y `error` descriptivo.

    NO inventes pedidos: si el JSON vuelve vacío o con error, informalo al usuario
    y pedí aclaraciones en lugar de adivinar. Esta herramienta es de SOLO LECTURA:
    no modifica ni elimina ningún dato.
    """
    pedidos = _FAKE_DB.get(cliente_id)
    if pedidos is None:
        return json.dumps(
            {"cliente_id": cliente_id, "pedidos": [], "error": f"Cliente {cliente_id} no encontrado."}
        )
    total = round(sum(p["total"] for p in pedidos), 2)
    return json.dumps(
        {
            "cliente_id": cliente_id,
            "cantidad_pedidos": len(pedidos),
            "monto_total": total,
            "pedidos": pedidos,
        }
    )


TOOLS = [buscar_pedidos]
