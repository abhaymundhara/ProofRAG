import json
import pytest
from unittest.mock import patch, MagicMock
from proofrag.generation.ollama import OllamaGenerator

def test_ollama_generator_payload():
    generator = OllamaGenerator(model="test-model", temperature=0.5, max_tokens=100)
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"response": "test output"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = generator.generate("hello")
        
        assert output == "test output"
        
        # Verify payload
        args, kwargs = mock_urlopen.call_args
        req = args[0]
        assert req.get_full_url() == "http://localhost:11434/api/generate"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "test-model"
        assert payload["prompt"] == "hello"
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 100

def test_ollama_connection_error():
    generator = OllamaGenerator()
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        from urllib.error import URLError
        mock_urlopen.side_effect = URLError(reason=ConnectionRefusedError())
        
        with pytest.raises(RuntimeError) as excinfo:
            generator.generate("hello")
        assert "not reachable" in str(excinfo.value)

def test_ollama_model_not_found():
    generator = OllamaGenerator(model="missing-model")
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        from urllib.error import HTTPError
        # 404 with "not found" body
        mock_error = HTTPError(url="http://localhost", code=404, msg="Not Found", hdrs={}, fp=MagicMock())
        mock_error.read = MagicMock(return_value=b"model 'missing-model' not found")
        mock_urlopen.side_effect = mock_error
        
        with pytest.raises(RuntimeError) as excinfo:
            generator.generate("hello")
        assert "ollama pull missing-model" in str(excinfo.value)

def test_ollama_bad_response_format():
    generator = OllamaGenerator()
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"error": "something went wrong"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        with pytest.raises(RuntimeError) as excinfo:
            generator.generate("hello")
        assert "missing 'response' field" in str(excinfo.value)
