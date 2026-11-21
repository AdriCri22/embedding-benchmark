"""
Generation interface for the embedding benchmark suite.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class GenerationInterface(ABC):
    """Interface for LLM generation and judging."""
    
    @abstractmethod
    def generate_answer(
        self, 
        model_cfg: Dict[str, Any], 
        query: str, 
        context_chunks: List[str]
    ) -> str:
        """Generate answer using LLM local via llama.cpp /v1/chat/completions."""
        pass
    
    @abstractmethod
    def judge_answer(
        self, 
        judge_cfg: Dict[str, Any], 
        query: str, 
        answer: str, 
        expected_answer: str, 
        lang: str
    ) -> Tuple[int, str]:
        """Evaluate answer with LLM judge. Returns (score, reasoning)."""
        pass