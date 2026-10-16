"""Herramientas del agente: simulan una base de datos empresarial de ventas y pedidos.

El LLM decide usarlas con total autonomía basándose EXCLUSIVAMENTE en sus docstrings
y esquemas de Pydantic. No existen rutas manuales if/else.
Aplica el principio de Least Privilege: herramientas de solo lectura y validación estricta.
"""
from __future__ import annotations

import json
from typing import Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# --- Base de datos relacional simulada (solo lectura) ---
_CLIENTES_DB: dict[int, dict[str, Any]] = {
    102: {
        "nombre": "TechCorp Argentina",
        "email": "compras@techcorp.com.ar",
        "pedidos": ["P-9001", "P-9144", "P-9277"],
    },
    205: {
        "nombre": "Logística del Sur",
        "email": "operaciones@logdelsur.com",
        "pedidos": ["P-8100"],
    },
}

_PEDIDOS_DB: dict[str, dict[str, Any]] = {
    "P-9001": {
        "pedido_id": "P-9001",
        "cliente_id": 102,
        "fecha": "2026-03-02",
        "total": 4200.0,
        "items": ["Licencia RedHat Enterprise (1 año)", "Soporte Premium"],
        "direccion_entrega": "Av. Corrientes 1234, CABA",
        "estado": "Entregado",
    },
    "P-9144": {
        "pedido_id": "P-9144",
        "cliente_id": 102,
        "fecha": "2026-05-19",
        "total": 5300.0,
        "items": ["4x Memoria RAM DDR5 64GB ECC", "2x SSD NVMe 2TB Enterprise"],
        "direccion_entrega": "Av. Corrientes 1234, CABA",
        "estado": "Entregado",
    },
    "P-9277": {
        "pedido_id": "P-9277",
        "cliente_id": 102,
        "fecha": "2026-09-01",
        "total": 5000.0,
        "items": ["Servidor Rack Dell PowerEdge R750 (1U)", "Kit de rieles deslizantes"],
        "direccion_entrega": "Datacenter Central, Parque Patricios, CABA",
        "estado": "Entregado",
    },
    "P-8100": {
        "pedido_id": "P-8100",
        "cliente_id": 205,
        "fecha": "2026-01-11",
        "total": 1200.0,
        "items": ["Switch Administrable Cisco Catalyst 24 puertos"],
        "direccion_entrega": "Ruta 3 Km 45, La Matanza",
        "estado": "Entregado",
    },
}


# ============================================================================
# Esquemas Pydantic para validación estricta de entradas
# ============================================================================

class BuscarPedidosInput(BaseModel):
    """Contrato de datos de entrada para la búsqueda de pedidos de un cliente."""

    cliente_id: int = Field(
        ...,
        ge=1,
        le=999999,
        description="Identificador numérico único del cliente (por ejemplo: 102 o 205). Debe ser un entero positivo.",
    )


class DetallePedidoInput(BaseModel):
    """Contrato de datos de entrada para consultar el detalle de un pedido específico."""

    pedido_id: str = Field(
        ...,
        min_length=3,
        max_length=20,
        description="Código alfanumérico del pedido con formato 'P-XXXX' (por ejemplo: 'P-9277' o 'P-9001').",
    )


# ============================================================================
# Herramientas decoradas con @tool y docstrings semánticos exhaustivos
# ============================================================================

@tool(args_schema=BuscarPedidosInput)
def buscar_pedidos(cliente_id: int) -> str:
    """Consulta el historial consolidado de pedidos de un cliente en la base de datos de ventas.

    CUÁNDO USAR ESTA HERRAMIENTA:
    - Utilízala cuando el usuario consulte cuántos pedidos realizó un cliente específico.
    - Utilízala cuando pregunten el monto total gastado o acumulado por un cliente.
    - Utilízala para listar los pedidos de un cliente antes de pedir el detalle de alguno de ellos.
    - Ejemplos de prompts: '¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?', '¿Qué compras hizo el cliente 205?'.

    PARÁMETROS DE ENTRADA:
    - cliente_id (int): ID numérico del cliente a consultar (ej: 102).

    SALIDA DEVUELTA:
    - String en formato JSON con la siguiente estructura:
      {
        "cliente_id": int,
        "nombre_cliente": str,
        "cantidad_pedidos": int,
        "monto_total": float,
        "pedidos": [
          {"pedido_id": str, "fecha": str, "total": float}
        ]
      }
    - Si el cliente no existe, retorna {"cliente_id": int, "error": "Cliente no encontrado", "pedidos": []}.

    REGLAS DE OPERACIÓN:
    - Esta herramienta es de SOLO LECTURA: no modifica datos.
    - Si la herramienta devuelve un error o lista vacía, no inventes información; informale al usuario de manera precisa.
    """
    cliente = _CLIENTES_DB.get(cliente_id)
    if cliente is None:
        return json.dumps(
            {
                "cliente_id": cliente_id,
                "error": f"Cliente {cliente_id} no encontrado en el sistema.",
                "pedidos": [],
            },
            ensure_ascii=False,
        )

    resumen_pedidos = []
    total_acumulado = 0.0
    for pid in cliente["pedidos"]:
        pdata = _PEDIDOS_DB.get(pid)
        if pdata:
            resumen_pedidos.append(
                {
                    "pedido_id": pdata["pedido_id"],
                    "fecha": pdata["fecha"],
                    "total": pdata["total"],
                }
            )
            total_acumulado += pdata["total"]

    return json.dumps(
        {
            "cliente_id": cliente_id,
            "nombre_cliente": cliente["nombre"],
            "cantidad_pedidos": len(resumen_pedidos),
            "monto_total": round(total_acumulado, 2),
            "pedidos": resumen_pedidos,
        },
        ensure_ascii=False,
    )


@tool(args_schema=DetallePedidoInput)
def obtener_detalle_pedido(pedido_id: str) -> str:
    """Consulta la información detallada de un pedido específico por su ID alfanumérico.

    CUÁNDO USAR ESTA HERRAMIENTA:
    - Utilízala cuando necesites conocer los productos específicos (items) contenidos en un pedido.
    - Utilízala cuando pregunten la dirección de entrega, estado del envío o desglose completo de un pedido.
    - Esta herramienta complementa a 'buscar_pedidos': primero buscas los pedidos del cliente y luego
      usas el ID del pedido (ej: 'P-9277') para obtener sus detalles.

    PARÁMETROS DE ENTRADA:
    - pedido_id (str): Identificador del pedido (ejemplo: 'P-9277', 'P-9001', 'P-8100').

    SALIDA DEVUELTA:
    - String en formato JSON con la siguiente estructura:
      {
        "pedido_id": str,
        "cliente_id": int,
        "fecha": str,
        "total": float,
        "items": list[str],
        "direccion_entrega": str,
        "estado": str
      }
    - Si el pedido no existe, retorna {"pedido_id": str, "error": "Pedido no encontrado"}.

    REGLAS DE OPERACIÓN:
    - Herramienta de SOLO LECTURA.
    - Respeta estrictamente los datos retornados.
    """
    clean_id = pedido_id.strip().upper()
    pedido = _PEDIDOS_DB.get(clean_id)
    if pedido is None:
        return json.dumps(
            {
                "pedido_id": clean_id,
                "error": f"Pedido '{clean_id}' no encontrado en el sistema.",
            },
            ensure_ascii=False,
        )

    return json.dumps(pedido, ensure_ascii=False)


# Lista exportada de herramientas disponibles para el LLM
TOOLS = [buscar_pedidos, obtener_detalle_pedido]
