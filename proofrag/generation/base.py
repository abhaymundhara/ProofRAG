from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    """Abstract base class for all generator backends."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a text completion for the given prompt."""
        pass
