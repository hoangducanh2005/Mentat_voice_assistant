import sys
from pathlib import Path

# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add src/ to python path so we can import glados
sys.path.append(str(Path(__file__).parent.parent / "src"))

from glados.engine import GladosConfig
from glados.ASR.asr import AudioTranscriber

def test_corrections():
    print("=== Testing ASR Post-Correction Layer ===")
    
    # 1. Load configuration and parse asr_corrections
    config = GladosConfig.from_yaml("configs/glados_config.yaml")
    corrections = config.asr_corrections
    print(f"Loaded {len(corrections.get('mappings', {}))} exact mappings and {len(corrections.get('target_words', []))} target words.")
    
    # 2. Instantiate transcriber manually with config corrections
    # We pass None or dummy paths for models because we are only testing the text correction method,
    # avoiding the overhead of loading real models/ONNX sessions.
    # To do this safely without loading models, we patch ort.InferenceSession or just mock it.
    from unittest.mock import MagicMock
    import onnxruntime as ort
    
    original_session = ort.InferenceSession
    ort.InferenceSession = MagicMock()
    
    try:
        transcriber = AudioTranscriber(
            model_path=Path("dummy_model"),
            tokens_file=Path("models/ASR/nemo-parakeet_tdt_ctc_110m_tokens.txt"),
            asr_corrections=corrections
        )
        
        # Test cases: (Input string from ASR -> Expected corrected output)
        test_cases = [
            ("tell me about pulrades", "tell me about Atreides"),
            ("tell me about pulrades and melang", "tell me about Atreides and Melange"),
            ("who is the kly sach", "who is the Kwisatz Haderach"),
            ("what is a benny gesserit", "what is a Bene Gesserit"),
            ("what is benny jesuit", "what is Bene Gesserit"),
            ("explain the kwisatz", "explain the Kwisatz Haderach"),
            ("tell me about arrakis and Fremen", "tell me about Arrakis and Fremen"),
            ("is jessica a bene gesserit", "is Jessica a Bene Gesserit"),
            ("what is a sietch and shai hulud", "what is a Sietch and Shai-Hulud"),
            ("tell me about sand worm", "tell me about Sandworm"),
        ]
        
        failures = 0
        for idx, (inp, expected) in enumerate(test_cases):
            out = transcriber.correct_transcription(inp)
            if out == expected:
                print(f"✅ Test {idx+1} Passed: '{inp}' -> '{out}'")
            else:
                print(f"❌ Test {idx+1} FAILED: '{inp}' -> expected '{expected}', got '{out}'")
                failures += 1
                
        if failures == 0:
            print("\n🎉 All ASR post-correction test cases passed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ {failures} test cases failed.")
            sys.exit(1)
            
    finally:
        ort.InferenceSession = original_session

if __name__ == "__main__":
    test_corrections()
