import os
import sys
import copy
from pathlib import Path
import requests
import yaml

# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add src/ to python path so we can import glados
sys.path.append(str(Path(__file__).parent.parent / "src"))
from glados.utils.rag_store import RagStore

def load_config():
    config_path = Path("configs/glados_config.yaml")
    if not config_path.exists():
        print(f"❌ Configuration file not found at {config_path}")
        sys.exit(1)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("Glados", {})
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        sys.exit(1)

def convert_messages_for_gemini(messages):
    gemini_contents = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            gemini_contents.append({
                "role": "user",
                "parts": [{"text": f"System: {content}"}]
            })
        elif role == "user":
            gemini_contents.append({
                "role": "user", 
                "parts": [{"text": content}]
            })
        elif role == "assistant":
            gemini_contents.append({
                "role": "model",
                "parts": [{"text": content}]
            })
    return gemini_contents

def main():
    print("==================================================")
    print("      DUNE Mentat Interactive Text Client")
    print("==================================================")
    
    config = load_config()
    api_type = config.get("api_type", "gemini")
    model = config.get("model", "gemini-2.0-flash")
    completion_url = str(config.get("completion_url", ""))
    
    # Resolve API Key
    api_key = config.get("api_key")
    if not api_key or api_key == "sk-xxx":
        env_var = "GEMINI_API_KEY" if api_type == "gemini" else "OPENAI_API_KEY"
        api_key = os.environ.get(env_var)
        
    if not api_key:
        print(f"❌ Error: API Key not found in config or environment variables.")
        sys.exit(1)
        
    # Initialize RagStore
    print("Loading vector store and initializing retrieval engine...")
    try:
        store = RagStore(
            config_path="configs/glados_config.yaml",
            vectors_path="data/vectors.npz"
        )
    except Exception as e:
        print(f"❌ Failed to load RagStore: {e}")
        sys.exit(1)
        
    # Build initial preprompt messages
    preprompt = []
    preprompt_config = config.get("personality_preprompt", [])
    for p in preprompt_config:
        # Pydantic validation handles this as system/user/assistant keys
        for role, content in p.items():
            preprompt.append({"role": role, "content": content})
            
    if not preprompt:
        preprompt = [{
            "role": "system", 
            "content": "You are a Mentat from the Dune universe. Speak logically and concisely."
        }]
        
    history = copy.deepcopy(preprompt)
    
    print("\nInitialization complete. Mentat is ready.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        try:
            query = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not query:
            continue
            
        if query.lower() in ["exit", "quit"]:
            print("Session ended.")
            break
            
        # 1. Search for relevant context
        print("🔍 Scanning archives...")
        retrieved = store.search(query, top_k=3)
        
        context_str = ""
        if retrieved:
            print(f"📄 Found {len(retrieved)} relevant reference documents.")
            context_str = "\n".join([f"- From {c['source']}: {c['text']}" for c in retrieved])
            for idx, r in enumerate(retrieved):
                print(f"   [{idx+1}] Score: {r['score']:.4f} | {r['source']}")
        else:
            print("📄 No relevant reference documents found in database.")
            
        # 2. Append query to history (clean version)
        history.append({"role": "user", "content": query})
        
        # 3. Build messages with context for current call
        messages_for_llm = copy.deepcopy(history)
        if context_str:
            messages_for_llm[-1]["content"] = (
                f"Context from Mentat Archives:\n{context_str}\n\n"
                f"Query: {query}"
            )
            
        # 4. Prepare API request
        headers = {"Content-Type": "application/json"}
        if api_type == "gemini":
            # Strip model name from completion URL or construct it
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": convert_messages_for_gemini(messages_for_llm),
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1024
                }
            }
        else:
            if "chat/completions" in completion_url:
                url = completion_url
            else:
                url = "https://api.openai.com/v1/chat/completions"
            headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": model,
                "messages": messages_for_llm
            }
            
        # 5. Call API
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            res_data = response.json()
            
            # Extract content
            response_text = ""
            if api_type == "gemini":
                if "candidates" in res_data and len(res_data["candidates"]) > 0:
                    parts = res_data["candidates"][0].get("content", {}).get("parts", [])
                    if parts:
                        response_text = parts[0].get("text", "")
            else:
                if "choices" in res_data and len(res_data["choices"]) > 0:
                    response_text = res_data["choices"][0].get("message", {}).get("content", "")
                    
            if response_text:
                print(f"\nMentat > {response_text}\n")
                # Append assistant response to history
                history.append({"role": "assistant", "content": response_text})
            else:
                print(f"\n⚠️ Mentat returned empty response. Response data: {res_data}\n")
                
        except Exception as e:
            print(f"\n❌ Error calling LLM API: {e}\n")

if __name__ == "__main__":
    main()
