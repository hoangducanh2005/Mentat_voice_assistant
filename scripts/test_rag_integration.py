import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add src/ to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from glados.engine import Glados, GladosConfig

def test_integration():
    print("=== Testing RAG Integration ===")
    
    # 1. Load config
    config = GladosConfig.from_yaml("configs/glados_config.yaml")
    
    # 2. Mock heavy ML models and sounddevice to avoid starting threads/audio streams
    mock_asr = MagicMock()
    mock_vad = MagicMock()
    mock_tts = MagicMock()
    mock_tts.sample_rate = 24000
    mock_tts.synthesize_audio.return_value = []
    
    # Mock sounddevice functions used in engine.py
    import sounddevice as sd
    sd.InputStream = MagicMock()
    sd.OutputStream = MagicMock()
    sd.play = MagicMock()
    sd.stop = MagicMock()
    sd.wait = MagicMock()
    
    # 3. Intercept LLM HTTP requests
    import requests
    original_post = requests.post
    
    captured_payload = None
    
    class MockResponse:
        def __init__(self):
            self.headers = {"content-type": "application/json"}
            self.status_code = 200
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def raise_for_status(self):
            pass
        def json(self):
            # Return a mock Gemini completion response
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "The Kwisatz Haderach is the ultimate Bene Gesserit breeding goal."
                                }
                            ]
                        }
                    }
                ]
            }
            
    def mock_post(url, headers=None, json=None, stream=False, timeout=None):
        nonlocal captured_payload
        captured_payload = json
        print(f"\n[HTTP POST] Intercepted call to completion URL: {url}")
        return MockResponse()
        
    requests.post = mock_post
    
    try:
        # Create glados engine
        glados = Glados(
            asr_model=mock_asr,
            tts_model=mock_tts,
            vad_model=mock_vad,
            completion_url=str(config.completion_url),
            model=config.model,
            api_key=config.api_key,
            interruptible=config.interruptible,
            wake_word=config.wake_word,
            announcement=None, # bypass announcement synthesis
            personality_preprompt=tuple(config.to_chat_messages()),
            api_type=config.api_type,
            rag_enabled=config.rag_enabled,
            rag_top_k=config.rag_top_k
        )
        
        # Verify RAG configuration was correctly loaded and store initialized
        print(f"RAG Enabled: {glados.rag_enabled}")
        print(f"RAG Top-K: {glados.rag_top_k}")
        assert glados.rag_enabled is True
        assert glados.rag_store is not None
        
        # Mock the get_query_embedding method of RagStore to return the first stored embedding
        # This prevents making real HTTP calls to Gemini API for embeddings during this test
        if glados.rag_store.embeddings.size > 0:
            glados.rag_store.get_query_embedding = MagicMock(
                return_value=glados.rag_store.embeddings[0]
            )
            print("Mocked RagStore.get_query_embedding with first vector from database.")
        
        # 4. Trigger query processing
        query = "Who is the Kwisatz Haderach?"
        print(f"\nPutting query in queue: '{query}'")
        glados.llm_queue.put(query)
        
        # Give the background thread a short moment to process
        time.sleep(2.0)
        
        # 5. Assertions
        assert captured_payload is not None, "Failed to capture LLM API request payload."
        print("\nCaptured API Payload sent to LLM:")
        import json as json_mod
        print(json_mod.dumps(captured_payload, indent=2))
        
        # Check that the payload contains retrieved context
        gemini_contents = captured_payload.get("contents", [])
        assert len(gemini_contents) > 0
        last_msg = gemini_contents[-1]
        last_text = last_msg["parts"][0]["text"]
        
        assert "Context from Mentat Archives:" in last_text
        assert "Kwisatz Haderach" in last_text
        print("\n✅ Verification Successful: Context successfully retrieved from RagStore and injected into LLM request!")
        
        # Clean shutdown of threads
        glados.shutdown_event.set()
        
    finally:
        requests.post = original_post

if __name__ == "__main__":
    test_integration()
