import os
import sys
import yaml
import requests
import numpy as np
from pathlib import Path

# Cấu hình stdout sử dụng UTF-8 để tránh lỗi hiển thị Unicode trên Windows console
sys.stdout.reconfigure(encoding='utf-8')

CONFIG_PATH = Path("configs/glados_config.yaml")
KNOWLEDGE_DIR = Path("data/knowledge")
OUTPUT_PATH = Path("data/vectors.npz")

def load_config():
    """Load configuration from glados_config.yaml"""
    if not CONFIG_PATH.exists():
        print(f"❌ Không tìm thấy file cấu hình tại: {CONFIG_PATH}")
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("Glados", {})
    except Exception as e:
        print(f"❌ Lỗi đọc cấu hình: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks of approx chunk_size characters.
    Adjusts boundaries to nearest spaces to prevent word splitting.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:].strip())
            break
            
        # Try to find a whitespace near the boundary to avoid cutting words
        # Search backwards up to 30 characters
        space_idx = text.rfind(' ', end - 30, end)
        if space_idx != -1:
            end = space_idx
            
        chunks.append(text[start:end].strip())
        start = end - overlap
        
    return [c for c in chunks if c]

def load_documents():
    """Scan knowledge dir and chunk all documents"""
    if not KNOWLEDGE_DIR.exists():
        print(f"❌ Thư mục tri thức không tồn tại: {KNOWLEDGE_DIR}")
        return []
        
    documents = []
    print(f"Scanning knowledge directory: {KNOWLEDGE_DIR}...")
    for file_path in KNOWLEDGE_DIR.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    continue
                chunks = chunk_text(content)
                print(f"📄 Loaded {file_path.name}: Generated {len(chunks)} chunks.")
                for idx, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "source": file_path.name,
                        "chunk_id": idx
                    })
        except Exception as e:
            print(f"❌ Không thể đọc file {file_path.name}: {e}")
            
    return documents

def get_embeddings(texts, api_type, api_key, completion_url):
    """
    Generate embeddings for a list of texts using the configured API.
    """
    import time
    embeddings = []
    headers = {"Content-Type": "application/json"}
    
    # Batch processing (using 15 texts per request to stay under the 100 RPM free tier quota)
    batch_size = 15
    total_texts = len(texts)
    
    # Set up URL and headers based on API type
    if api_type == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={api_key}"
    else:
        # OpenAI style: Rewrite chat/completions to embeddings
        if "chat/completions" in completion_url:
            url = completion_url.replace("chat/completions", "embeddings")
        else:
            url = "https://api.openai.com/v1/embeddings"
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"Embedding API Endpoint: {url}")
    
    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"Generating embeddings for batch {i//batch_size + 1}/{(total_texts-1)//batch_size + 1}...")
        
        # Retry logic with exponential backoff on 429
        max_attempts = 5
        attempt = 0
        success = False
        
        while attempt < max_attempts and not success:
            try:
                if api_type == "gemini":
                    requests_list = []
                    for text in batch_texts:
                        requests_list.append({
                            "model": "models/gemini-embedding-001",
                            "content": {"parts": [{"text": text}]}
                        })
                    payload = {"requests": requests_list}
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                else:
                    # OpenAI style allows batch embedding in a single request
                    payload = {
                        "model": "text-embedding-3-small",
                        "input": batch_texts
                    }
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                # Check for rate limits (429)
                if response.status_code == 429:
                    print(f"⚠️ Exceeded rate limit (429). Attempt {attempt + 1}/{max_attempts}. Sleeping 60s...")
                    time.sleep(60)
                    attempt += 1
                    continue
                    
                response.raise_for_status()
                res_data = response.json()
                
                if api_type == "gemini":
                    for item in res_data.get("embeddings", []):
                        vector = item.get("values", [])
                        if vector:
                            embeddings.append(vector)
                        else:
                            raise ValueError(f"Unexpected item in response: {item}")
                else:
                    data_list = res_data.get("data", [])
                    data_list.sort(key=lambda x: x.get("index", 0))
                    for item in data_list:
                        embeddings.append(item.get("embedding"))
                
                success = True
                
            except Exception as e:
                # Catch rate limiting exceptions
                is_rate_limit = False
                if isinstance(e, requests.RequestException) and hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code == 429:
                        is_rate_limit = True
                        print(f"⚠️ Exceeded rate limit (429). Attempt {attempt + 1}/{max_attempts}. Sleeping 60s...")
                        time.sleep(60)
                        attempt += 1
                
                if not is_rate_limit:
                    print(f"❌ Error generating embeddings for batch starting at {i}: {e}")
                    if isinstance(e, requests.RequestException) and hasattr(e, 'response') and e.response is not None:
                        print(f"Response details: {e.response.text}")
                    return None
                    
        if not success:
            print(f"❌ Failed to generate embeddings for batch starting at {i} after {max_attempts} attempts.")
            return None
            
        # Stagger requests to stay under 100 RPM (15 requests per minute -> ~1 request per 4 seconds)
        if i + batch_size < total_texts:
            time.sleep(10)
            
    return embeddings

def main():
    print("🚀 Khởi động tiến trình Ingestion (Số hóa tri thức)...")
    
    # 1. Load config
    config = load_config()
    if not config:
        sys.exit(1)
        
    api_type = config.get("api_type", "openai")
    api_key = config.get("api_key")
    completion_url = str(config.get("completion_url", ""))
    
    # Fallback to env variables if config holds placeholder
    if not api_key or api_key == "sk-xxx":
        print("⚠️ Phát hiện API Key trống hoặc là placeholder. Đang tìm trong biến môi trường...")
        if api_type == "gemini":
            env_key = os.environ.get("GEMINI_API_KEY")
        else:
            env_key = os.environ.get("OPENAI_API_KEY")
            
        if env_key:
            api_key = env_key
        else:
            if not api_key:
                print("❌ Lỗi: Không tìm thấy API Key hợp lệ trong cả cấu hình lẫn biến môi trường!")
                sys.exit(1)
            else:
                print("⚠️ Không tìm thấy key trong biến môi trường. Tiếp tục sử dụng key mặc định từ cấu hình...")
        
    # 2. Load and chunk documents
    documents = load_documents()
    if not documents:
        print("❌ Không có tài liệu nào để xử lý.")
        sys.exit(1)
        
    print(f"Tổng số chunks đã được tạo: {len(documents)}")
    
    # 3. Generate embeddings
    texts = [doc["text"] for doc in documents]
    embeddings_list = get_embeddings(texts, api_type, api_key, completion_url)
    
    if not embeddings_list or len(embeddings_list) != len(texts):
        print("❌ Quá trình tạo vector nhúng (embeddings) thất bại hoặc không đầy đủ.")
        sys.exit(1)
        
    # 4. Save to numpy vectors.npz
    embeddings_arr = np.array(embeddings_list, dtype=np.float32)
    
    # Store text and metadata alongside embeddings
    texts_arr = np.array(texts, dtype=object)
    sources_arr = np.array([doc["source"] for doc in documents], dtype=object)
    
    print(f"Saving vectors to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        embeddings=embeddings_arr,
        texts=texts_arr,
        sources=sources_arr
    )
    
    print(f"✅ Ingestion hoàn tất thành công! Đã lưu {len(texts)} vectors với chiều kích thước: {embeddings_arr.shape}")

if __name__ == "__main__":
    main()
