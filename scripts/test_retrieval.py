import sys
from pathlib import Path

# Reconfigure stdout to use UTF-8 to prevent unicode encode errors on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add src/ to Python path so we can import glados
sys.path.append(str(Path(__file__).parent.parent / "src"))

from glados.utils.rag_store import RagStore

def main():
    print("=== Testing RagStore Retrieval Engine ===")
    
    # Initialize the RagStore
    # By default, it will load configs/glados_config.yaml and data/vectors.npz
    try:
        store = RagStore(
            config_path="configs/glados_config.yaml",
            vectors_path="data/vectors.npz"
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize RagStore: {e}")
        sys.exit(1)
        
    if store.embeddings.size == 0:
        print("[ERROR] RagStore was initialized, but vector database is empty or not loaded.")
        sys.exit(1)
        
    # Test queries
    queries = [
        "Tell me about the Kwisatz Haderach",
        "What is Melange?",
        "Who is Paul Atreides?"
    ]
    
    for query in queries:
        print(f"\n[QUERY] '{query}'")
        try:
            results = store.search(query, top_k=3)
            if not results:
                print("[WARN] No results found or search failed.")
                continue
                
            for idx, r in enumerate(results):
                print(f"  [{idx+1}] Score: {r['score']:.4f} | Source: {r['source']}")
                # Print first 200 characters of the text chunk
                snippet = r['text'].replace('\n', ' ')
                print(f"      Snippet: {snippet[:200]}...")
        except Exception as e:
            print(f"[ERROR] Error during search: {e}")

if __name__ == "__main__":
    main()

