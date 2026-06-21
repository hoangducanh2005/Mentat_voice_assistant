import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from glados.engine import Glados, GladosConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files for HTML/JS
static_dir = Path("data/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global Glados instance
glados = None
active_websockets: set[WebSocket] = set()
main_loop = None

def on_audio_out(audio: np.ndarray, text: str):
    """Callback when Glados has audio to play"""
    if not active_websockets or not main_loop:
        return
        
    audio_bytes = audio.tobytes()
    # Send text first as json, then audio as binary
    message = json.dumps({"type": "audio_text", "text": text})
    
    async def broadcast():
        for ws in list(active_websockets):
            try:
                await ws.send_text(message)
                await ws.send_bytes(audio_bytes)
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
                
    asyncio.run_coroutine_threadsafe(broadcast(), main_loop)

def on_state_change(state: str):
    if not main_loop:
        return
    message = json.dumps({"type": "state", "state": state})
    async def broadcast():
        for ws in list(active_websockets):
            try:
                await ws.send_text(message)
            except:
                pass
    asyncio.run_coroutine_threadsafe(broadcast(), main_loop)

def on_transcript(raw: str, clean: str):
    if not main_loop:
        return
    message = json.dumps({"type": "transcript", "raw": raw, "clean": clean})
    async def broadcast():
        for ws in list(active_websockets):
            try:
                await ws.send_text(message)
            except:
                pass
    asyncio.run_coroutine_threadsafe(broadcast(), main_loop)

@app.on_event("startup")
async def startup_event():
    global glados, main_loop
    main_loop = asyncio.get_running_loop()
    try:
        config = GladosConfig.from_yaml("configs/glados_config.yaml")
        # Initialize without mic/speaker
        glados = Glados.from_config(config, use_mic_speaker=False)
        glados.on_audio_out = on_audio_out
        glados.on_state_change = on_state_change
        glados.on_transcript = on_transcript
        
        # We need to run the Glados event loop in a background thread
        import threading
        t = threading.Thread(target=glados.start_listen_event_loop, daemon=True)
        t.start()
        logger.info("Glados engine started in background thread")
    except Exception as e:
        logger.error(f"Failed to start Glados: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if glados:
        glados.shutdown_event.set()

@app.get("/")
async def get_index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Index.html not found in data/static/</h1>")

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        # Send initial config to client so they know what sample rate to expect
        await websocket.send_text(json.dumps({
            "type": "config",
            "tts_sample_rate": glados._tts.sample_rate if glados else 24000
        }))
        
        while True:
            # We expect either JSON messages or Binary audio chunks
            message = await websocket.receive()
            if "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "ping":
                    pass # Keepalive
            elif "bytes" in message:
                if glados:
                    # Client should send Float32 PCM at glados.SAMPLE_RATE (16000)
                    audio_data = np.frombuffer(message["bytes"], dtype=np.float32)
                    glados.push_audio_chunk(audio_data)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
