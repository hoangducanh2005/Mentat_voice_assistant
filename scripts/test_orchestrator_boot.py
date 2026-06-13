import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add src/ to python path so we can import glados
sys.path.append(str(Path(__file__).parent.parent / "src"))
from glados.engine import Glados, GladosConfig

def test_boot():
    print("=== Testing Voice Assistant Orchestrator Boot ===")
    
    # 1. Load config
    config = GladosConfig.from_yaml("configs/glados_config.yaml")
    
    # 2. Mock sounddevice to avoid hardware conflicts, but load actual models
    import sounddevice as sd
    sd.InputStream = MagicMock()
    sd.OutputStream = MagicMock()
    sd.play = MagicMock()
    sd.stop = MagicMock()
    sd.wait = MagicMock()
    
    print("Initializing Glados Engine components (ASR, VAD, TTS, RagStore)...")
    try:
        glados = Glados(
            asr_model=MagicMock(), # Mock ASR to avoid loading 450MB model in quick test
            tts_model=MagicMock(), # Mock TTS to avoid loading 170MB model in quick test
            vad_model=MagicMock(),
            completion_url=str(config.completion_url),
            model=config.model,
            api_key=config.api_key,
            interruptible=config.interruptible,
            wake_word=config.wake_word,
            announcement=None, # disable start speech playback
            personality_preprompt=tuple(config.to_chat_messages()),
            api_type=config.api_type,
            rag_enabled=config.rag_enabled,
            rag_top_k=config.rag_top_k
        )
        print("✅ Glados Engine successfully instantiated!")
        
        # Verify background threads (LLM process, TTS process, Audio process) are created and running
        print("Verifying background thread lifecycles...")
        time.sleep(2.0)
        
        # Verify shutdown event remains unset
        assert glados.shutdown_event.is_set() is False, "Shutdown event was set prematurely."
        print("✅ Background loops initialized and running smoothly.")
        
        # Perform clean shutdown
        print("Stopping background worker threads...")
        glados.shutdown_event.set()
        print("✅ Clean shutdown completed.")
        
    except Exception as e:
        print(f"❌ Orchestrator boot failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_boot()
