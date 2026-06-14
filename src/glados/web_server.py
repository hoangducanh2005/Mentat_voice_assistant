import asyncio
import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

from .engine import Glados

app = FastAPI(title="Mentat Voice Assistant Cockpit")

# Global reference to glados instance
glados_instance: Glados = None
current_state = "IDLE"

class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict[str, Any]) -> None:
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

    def broadcast_json_sync(self, data: dict[str, Any]) -> None:
        if self.loop and self.active_connections:
            asyncio.run_coroutine_threadsafe(self.broadcast_json(data), self.loop)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    manager.set_loop(asyncio.get_running_loop())
    await manager.connect(websocket)
    # Immediately send current state on connection
    await websocket.send_json({"type": "state", "data": current_state})
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                action_type = payload.get("type")
                if action_type == "manual_trigger" and glados_instance:
                    glados_instance.start_manual_listening()
                elif action_type == "reset" and glados_instance:
                    glados_instance.reset()
                elif action_type == "text_query" and glados_instance:
                    query = payload.get("data")
                    if query:
                        logger.info(f"Dashboard text query received: '{query}'")
                        glados_instance.llm_queue.put(query)
                        glados_instance.processing = True
                        glados_instance.currently_speaking.set()
            except Exception as e:
                logger.error(f"Error handling WS message: {e}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def setup_glados_callbacks(glados: Glados) -> None:
    global glados_instance, current_state
    glados_instance = glados
    current_state = "IDLE"

    def on_state_change(state: str) -> None:
        global current_state
        current_state = state
        manager.broadcast_json_sync({"type": "state", "data": state})

    def on_transcript(raw: str, corrected: str) -> None:
        manager.broadcast_json_sync({"type": "transcript", "data": {"raw": raw, "corrected": corrected}})

    def on_assistant(sentence: str) -> None:
        manager.broadcast_json_sync({"type": "assistant_sentence", "data": sentence})

    def on_assistant_chunk(chunk: str) -> None:
        manager.broadcast_json_sync({"type": "assistant_chunk", "data": chunk})

    def on_rag(chunks: list[dict[str, Any]]) -> None:
        manager.broadcast_json_sync({"type": "rag", "data": chunks})

    def loguru_sink(message: Any) -> None:
        record = message.record
        log_payload = {
            "time": record["time"].strftime("%H:%M:%S.%f")[:-3],
            "level": record["level"].name,
            "message": record["message"],
            "name": record["name"],
            "line": record["line"]
        }
        manager.broadcast_json_sync({"type": "log", "data": log_payload})

    glados.on_state_change = on_state_change
    glados.on_transcript = on_transcript
    glados.on_assistant = on_assistant
    glados.on_assistant_chunk = on_assistant_chunk
    glados.on_rag = on_rag

    # Register Loguru sink to intercept logs for dashboard telemetry
    logger.add(loguru_sink, format="{message}", level="DEBUG")

# Mount web assets
web_dir = Path(__file__).parent / "web"
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

def start_dashboard_server(glados: Glados, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Configures the callbacks on the Glados instance and runs the FastAPI server."""
    setup_glados_callbacks(glados)
    logger.success(f"Dashboard Web Server initializing on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
