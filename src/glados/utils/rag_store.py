import os
from pathlib import Path
import numpy as np
import requests
import yaml

class RagStore:
    def __init__(
        self, 
        config_path: str | Path = "configs/glados_config.yaml", 
        vectors_path: str | Path = "data/vectors.npz"
    ):
        self.config_path = Path(config_path)
        self.vectors_path = Path(vectors_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize vector store variables
        self.embeddings = np.array([], dtype=np.float32)
        self.texts = np.array([], dtype=object)
        self.sources = np.array([], dtype=object)
        self._load_vectors()
        
        # API configuration
        self.api_type = self.config.get("api_type", "openai")
        self.completion_url = str(self.config.get("completion_url", ""))
        self.api_key = self._resolve_api_key()

    def _load_config(self) -> dict:
        """Loads glados config with absolute path fallback."""
        if not self.config_path.exists():
            # Fallback for when running from a subfolder, e.g. tests/
            fallback_path = Path(__file__).parents[3] / self.config_path
            if fallback_path.exists():
                self.config_path = fallback_path
                
        if not self.config_path.exists():
            print(f"[WARN] RagStore: Configuration file not found at {self.config_path}. Using empty config.")
            return {}
            
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("Glados", {})
        except Exception as e:
            print(f"[WARN] RagStore: Error reading configuration: {e}")
            return {}

    def _load_vectors(self):
        """Loads vectors and metadata from the compiled NPZ file."""
        if not self.vectors_path.exists():
            # Fallback for when running from a subfolder, e.g. tests/
            fallback_path = Path(__file__).parents[3] / self.vectors_path
            if fallback_path.exists():
                self.vectors_path = fallback_path
                
        if not self.vectors_path.exists():
            print(f"[WARN] RagStore: Vector database file not found at {self.vectors_path}.")
            return
            
        try:
            data = np.load(self.vectors_path, allow_pickle=True)
            self.embeddings = data["embeddings"].astype(np.float32)
            self.texts = data["texts"]
            self.sources = data["sources"]
            print(f"[INFO] RagStore: Successfully loaded {len(self.texts)} vectors from {self.vectors_path}")
        except Exception as e:
            print(f"[ERROR] RagStore: Error loading vector database: {e}")

    def _resolve_api_key(self) -> str | None:
        """Resolves the API key from config or environment variables."""
        key = self.config.get("api_key")
        if not key or key == "sk-xxx" or key.strip() == "":
            # Fallback to environment variables
            if self.api_type == "gemini":
                key = os.environ.get("GEMINI_API_KEY")
            else:
                key = os.environ.get("OPENAI_API_KEY")
        return key

    def get_query_embedding(self, query_text: str) -> np.ndarray:
        """Generates embedding vector for the query text using Gemini or OpenAI API."""
        if not self.api_key:
            raise ValueError(
                "API Key is not configured. Set 'api_key' in your config or the corresponding environment variable."
            )
            
        headers = {"Content-Type": "application/json"}
        
        if self.api_type == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent?key={self.api_key}"
            payload = {
                "model": "models/gemini-embedding-2",
                "content": {
                    "parts": [{"text": query_text}]
                }
            }
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            res_data = response.json()
            vector = res_data.get("embedding", {}).get("values", [])
            if not vector:
                raise ValueError(
                    f"Invalid Gemini response: 'embedding.values' not found. Response: {res_data}"
                )
            return np.array(vector, dtype=np.float32)
            
        else:
            # OpenAI style
            if "chat/completions" in self.completion_url:
                url = self.completion_url.replace("chat/completions", "embeddings")
            else:
                url = "https://api.openai.com/v1/embeddings"
                
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": "text-embedding-3-small",
                "input": query_text
            }
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            res_data = response.json()
            data_list = res_data.get("data", [])
            if not data_list:
                raise ValueError(
                    f"Invalid OpenAI response: 'data' list is empty. Response: {res_data}"
                )
            vector = data_list[0].get("embedding")
            if not vector:
                raise ValueError(
                    f"Invalid OpenAI response: 'embedding' not found in data[0]. Response: {res_data}"
                )
            return np.array(vector, dtype=np.float32)

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """Performs a NumPy vectorized Cosine Similarity search on the vector database."""
        if self.embeddings.size == 0 or len(self.texts) == 0:
            print("[WARN] RagStore: Search called but vector database is empty or not loaded.")
            return []
            
        try:
            query_vec = self.get_query_embedding(query_text)
        except Exception as e:
            print(f"[ERROR] RagStore: Error generating query embedding: {e}")
            return []
            
        # Cosine Similarity = dot(A, B) / (norm(A) * norm(B))
        dot_products = np.dot(self.embeddings, query_vec)
        norms_db = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(query_vec)
        
        # Use a small epsilon to avoid divide-by-zero errors
        similarities = dot_products / (norms_db * norm_query + 1e-10)
        
        # Sort in descending order of similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": str(self.texts[idx]),
                "source": str(self.sources[idx]),
                "score": float(similarities[idx])
            })
            
        return results
