import re
from typing import Set

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

def strip_citations(text: str) -> str:
    """Removes citations like [rec-0], [record_id=...], [1], (rec-0) from text."""
    if not text:
        return ""
    # Remove [rec-0], [record_id=...], [1]
    text = re.sub(r'\[(?:rec-\d+|record_id=[^\]]+|\d+)\]', '', text)
    # Remove (rec-0), (record_id=...), (1)
    text = re.sub(r'\((?:rec-\d+|record_id=[^)]+|\d+)\)', '', text)
    return text

def clean_model_answer(text: str) -> str:
    """Removes model-specific artifacts like 'thought' blocks and '<channel|>' tags."""
    if not text:
        return ""
        
    # 1. Handle <channel|> tags (common in Gemma/thinking models)
    if "<channel|>" in text:
        text = text.split("<channel|>")[-1]
        
    # 2. Handle 'thought\n' blocks
    if text.startswith("thought\n"):
        # If there's a clear separation after thought (e.g. double newline or some marker),
        # but usually <channel|> handled it. If not, we look for common patterns.
        # For now, if <channel|> wasn't there, we just strip the 'thought\n' prefix if it's the very start.
        text = text[len("thought\n"):]

    return text.strip()

def contains_gold_answer(generated_answer: str, gold_answer: str) -> bool:
    """Checks if the normalized gold answer is contained within the normalized generated answer.
    Handles citation stripping and multi-part 'and' gold answers.
    """
    if not gold_answer:
        return False
        
    # clean_model_answer should have been called before this, but we'll be safe
    gen_to_score = clean_model_answer(generated_answer)
        
    # 1. Strip citations from generated answer
    clean_gen = strip_citations(gen_to_score)
    
    # 2. Normalize both
    norm_gen = normalize_answer(clean_gen)
    norm_gold = normalize_answer(gold_answer)
    
    # 3. Direct substring check
    if norm_gold in norm_gen:
        return True
        
    # 4. Handle "and" cases (e.g. "Tom and Sarah")
    if " and " in gold_answer.lower():
        parts = re.split(r'\s+and\s+', gold_answer, flags=re.IGNORECASE)
        norm_parts = [normalize_answer(p) for p in parts if normalize_answer(p)]
        if norm_parts and all(p in norm_gen for p in norm_parts):
            return True
            
    return False

def extract_entities_from_answer(text: str) -> Set[str]:
    """Simple heuristic to extract capitalized words (potential entities) from text."""
    clean_text = strip_citations(clean_model_answer(text))
    entities = set(re.findall(r'\b[A-Z][a-z]+\b', clean_text))
    return entities
