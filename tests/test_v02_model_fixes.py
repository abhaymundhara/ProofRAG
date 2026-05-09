import json
from unittest.mock import patch, MagicMock
from proofrag.generation.ollama import OllamaGenerator
from proofrag.evaluation.answer_metrics import strip_citations, contains_gold_answer

def test_ollama_chat_payload():
    generator = OllamaGenerator(model="test-model", endpoint_mode="chat")
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        # Ollama /api/chat response format
        mock_response.read.return_value = json.dumps({
            "model": "test-model",
            "message": {"role": "assistant", "content": "hello world"},
            "done": True
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        output = generator.generate("hi")
        assert output == "hello world"
        
        args, kwargs = mock_urlopen.call_args
        req = args[0]
        assert "/api/chat" in req.get_full_url()
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["think"] is False
        assert payload["messages"][0]["content"] == "hi"

def test_generate_with_metadata_thinking():
    generator = OllamaGenerator(model="thinking-model")
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "message": {"content": "Final Answer"},
            "thinking": "I am thinking deeply..."
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        meta = generator.generate_with_metadata("test")
        assert meta["content"] == "Final Answer"
        assert meta["thinking"] == "I am thinking deeply..."

def test_strip_citations():
    assert strip_citations("Tom [rec-0] asked.") == "Tom  asked."
    assert strip_citations("Sarah [record_id=doc-1] and LiHua (rec-2)") == "Sarah  and LiHua "
    assert strip_citations("Check [1] and (2)") == "Check  and "

def test_improved_contains_gold_answer():
    # q006 case
    assert contains_gold_answer(
        "Tom [rec-0] and Sarah [rec-1] asked LiHua about the laptop warranty.",
        "Tom and Sarah"
    ) is True
    
    # Non-contiguous parts
    assert contains_gold_answer(
        "First Tom spoke, then later Sarah joined.",
        "Tom and Sarah"
    ) is True
    
    # Exact match still works
    assert contains_gold_answer("The answer is Tom.", "Tom") is True
    
    # Rejection works
    assert contains_gold_answer("Only Tom was there.", "Tom and Sarah") is False
    assert contains_gold_answer("I don't know.", "Tom") is False
