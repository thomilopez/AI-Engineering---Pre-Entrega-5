# Pre-entrega 5: Agente de Razonamiento Cíclico con Memoria Persistente

Agente ReAct con **LangGraph**: razona, llama a una tool que simula una base de
datos de pedidos, observa el resultado y responde. Memoria por `thread_id`
con `AsyncSqliteSaver` (archivo `checkpoints.db`).

## Estructura

```
.
├── tools.py            # @tool buscar_pedidos + esquema Pydantic + DB simulada
├── agent.py            # MessagesState, StateGraph (agent <-> ToolNode), tools_condition, bind_tools
├── main.py             # demo async multi-paso (2 turnos, mismo thread_id) + trace_react.json
├── trace_example.json  # ejemplo de traza ReAct esperada
├── requirements.txt
├── .env.example
└── README.md
```

Grafo: `START -> agent -> [tools_condition] -> tools -> agent ... -> END`.
El ciclo existe porque la arista `tools -> agent` devuelve el flujo al LLM.

## Levantar el entorno

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # completar OPENAI_API_KEY (nunca hardcodear keys)
```

Requiere Python 3.12+. Solo `OPENAI_API_KEY` (y opcional `MODEL_NAME`,
por defecto `gpt-4o-mini`).

## Ejecutar la demo

```bash
python main.py
```

1. Turno 1: `¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?`
   -> el agente invoca `buscar_pedidos(cliente_id=102)` y responde
   `3 pedidos por $14.500`.
2. Turno 2 (mismo `thread_id`): `¿Y el último?`
   -> responde `P-9277 ...` usando el checkpoint, sin que repitas el ID.
3. Guarda la traza real en `trace_react.json` e imprime el conteo de
   tool calls (esperado ≥ 2). Todo `.ainvoke` usa `recursion_limit=10`.

## Decisiones de diseño

| Punto | Por qué |
|---|---|
| `MessagesState` + `add_messages` | Los mensajes se anexan, jamás se sobrescriben |
| `tools_condition` | Sin `if/else` manual: el LLM decide solo cuándo usar tools |
| Docstring largo en la tool | El modelo elige la tool solo por su descripción |
| Pydantic `args_schema` | Valida `cliente_id` antes de tocar la DB |
| `AsyncSqliteSaver` + `thread_id` | Persistencia local que sobrevive reinicios |
| `recursion_limit=10` | Techo anti bucles infinitos y costos sorpresa |
| Nodos pequeños y async | Un nodo = una responsabilidad; no bloquea el loop |

## Traza esperada (resumen, ver `trace_example.json`)

```
Usuario: "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"
→ agente decide: buscar_pedidos(cliente_id=102)
→ tool devuelve: {"pedidos": 3, "total": 14500}
→ agente razona: ya tiene los datos → responde.
Respuesta: "El cliente 102 tuvo 3 pedidos por un total de $14.500."
(con el mismo thread_id, "¿y el último?" recuerda el contexto.)
```
