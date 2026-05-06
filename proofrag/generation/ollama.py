import json
import urllib.request
import urllib.error
from .base import BaseGenerator

class OllamaGenerator(BaseGenerator):
    """Generator backend using a local Ollama instance."""
    
    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float = 0.0,
        max_tokens: int = 256
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        """Sends a prompt to the Ollama /api/generate endpoint."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                
                if "response" not in res_data:
                    raise RuntimeError(f"Ollama response missing 'response' field: {res_data}")
                
                return res_data["response"]
                
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Often means model not found
                res_body = e.read().decode("utf-8")
                if "not found" in res_body.lower():
                    raise RuntimeError(
                        f"Model '{self.model}' not found in Ollama. Try: ollama pull {self.model}"
                    ) from e
            raise RuntimeError(f"Ollama HTTP error {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            if hasattr(e, "reason") and (isinstance(e.reason, ConnectionRefusedError) or "connection refused" in str(e.reason).lower()):
                raise RuntimeError(
                    f"Ollama is not reachable at {self.base_url}. Start it with `ollama serve`."
                ) from e
            raise RuntimeError(f"Ollama request failed: {e}") from e
