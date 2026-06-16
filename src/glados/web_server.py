import asyncio
import json
from pathlib import Path
import platform
import random
import threading
import time
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from loguru import logger
import psutil
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

    def on_latency_profile(profile_data: dict[str, Any]) -> None:
        manager.broadcast_json_sync({"type": "latency_profile", "data": profile_data})

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
    glados.on_latency_profile = on_latency_profile

    # Register Loguru sink to intercept logs for dashboard telemetry
    logger.add(loguru_sink, format="{message}", level="DEBUG")

# Simulated temperature state tracking
simulated_temp = 42.0

def get_cpu_temperature() -> float:
    global simulated_temp
    system = platform.system()
    if system == "Linux":
        try:
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.exists():
                return float(temp_path.read_text().strip()) / 1000.0
        except Exception:
            pass
        try:
            temps = psutil.sensors_temperatures()
            if "cpu_thermal" in temps:
                return temps["cpu_thermal"][0].current
            elif "coretemp" in temps:
                return temps["coretemp"][0].current
        except Exception:
            pass
        return 45.0
    
    elif system == "Windows":
        try:
            import wmi
            w = wmi.WMI(namespace="root\\wmi")
            temperature_info = w.MSAcpi_ThermalZoneTemperature()
            if temperature_info:
                kelvin = temperature_info[0].CurrentTemperature / 10.0
                celsius = kelvin - 273.15
                if 0 < celsius < 100:
                    return celsius
        except Exception:
            pass
        
        target_temp = 42.0
        if current_state in ("THINKING", "SPEAKING"):
            target_temp = 58.0 + random.uniform(-1.0, 1.0)
        else:
            target_temp = 42.0 + random.uniform(-0.5, 0.5)
        
        simulated_temp += (target_temp - simulated_temp) * 0.1
        return round(simulated_temp, 1)
    
    else:
        return 40.0

def hardware_telemetry_worker() -> None:
    import os
    try:
        process = psutil.Process(os.getpid())
    except Exception:
        process = None

    while True:
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            system_ram_total = mem.total
            system_ram_used = mem.used
            system_ram_percent = mem.percent
            
            process_rss = 0
            if process:
                try:
                    process_rss = process.memory_info().rss
                except Exception:
                    pass
            
            asr_loaded = False
            rag_loaded = False
            if glados_instance:
                if getattr(glados_instance, "_asr_model", None) is not None:
                    asr_loaded = True
                if getattr(glados_instance, "rag_store", None) is not None:
                    rag_loaded = True
            
            asr_mem = 220 * 1024 * 1024 if asr_loaded else 0
            rag_mem = 10 * 1024 * 1024 if rag_loaded else 0
            other_mem = max(0, process_rss - asr_mem - rag_mem)
            
            cpu_temp = get_cpu_temperature()
            
            payload = {
                "cpu_percent": cpu_percent,
                "system_ram_percent": system_ram_percent,
                "system_ram_used": system_ram_used,
                "system_ram_total": system_ram_total,
                "process_rss": process_rss,
                "asr_mem": asr_mem,
                "rag_mem": rag_mem,
                "other_mem": other_mem,
                "cpu_temp": cpu_temp
            }
            
            manager.broadcast_json_sync({"type": "hardware", "data": payload})
        except Exception as e:
            logger.error(f"Error in telemetry worker: {e}")
        time.sleep(1.5)

# Mount web assets
web_dir = Path(__file__).parent / "web"
app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

def start_dashboard_server(glados: Glados, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Configures the callbacks on the Glados instance and runs the FastAPI server."""
    setup_glados_callbacks(glados)
    
    # Start background hardware telemetry worker
    t = threading.Thread(target=hardware_telemetry_worker, daemon=True)
    t.start()
    
    logger.success(f"Dashboard Web Server initializing on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
