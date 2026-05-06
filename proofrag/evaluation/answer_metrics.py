import re

def normalize_answer(text: str) -> str:
    """Normalizes answer text by lowercasing, stripping punctuation, and collapsing whitespace."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Strip punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def contains_gold_answer(generated_answer: str, gold_answer: str) -> bool:
    """Checks if the normalized gold answer is contained within the normalized generated answer."""
    norm_gen = normalize_answer(generated_answer)
    norm_gold = normalize_answer(gold_answer)
    
    if not norm_gold:
        return False
        
    return norm_gold in norm_gen
