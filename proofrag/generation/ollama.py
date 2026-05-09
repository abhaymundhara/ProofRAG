import json
import urllib.request
import urllib.error
from typing import Dict, Any
from .base import BaseGenerator

class OllamaGenerator(BaseGenerator):
    """Generator backend using a local Ollama instance."""
    
    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        temperature: float = 0.0,
        max_tokens: int = 256,
        endpoint_mode: str = "chat" # "chat" or "generate"
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint_mode = endpoint_mode

    def generate(self, prompt: str) -> str:
        """Returns the generated text content from the model."""
        return self.generate_with_metadata(prompt)["content"]

    def generate_with_metadata(self, prompt: str) -> Dict[str, Any]:
        """Sends a prompt to Ollama and returns content + metadata (like thinking)."""
        if self.endpoint_mode == "chat":
            return self._call_chat(prompt)
        else:
            return self._call_generate(prompt)

    def _call_chat(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "think": False, # Prevent spending tokens on thinking in visible output
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        raw_res = self._post(url, payload)
        
        message = raw_res.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking") or raw_res.get("thinking")
        
        return {
            "content": content,
            "thinking": thinking,
            "raw": raw_res
        }

    def _call_generate(self, prompt: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        raw_res = self._post(url, payload)
        
        content = raw_res.get("response", "")
        thinking = raw_res.get("thinking")
        
        return {
            "content": content,
            "thinking": thinking,
            "raw": raw_res
        }

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data
                
        except urllib.error.HTTPError as e:
            if e.code == 404:
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
