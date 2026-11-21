"""
Data loader interface for the embedding benchmark suite.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any


class DataLoaderInterface(ABC):
    """Interface for loading test data."""
    
    @abstractmethod
    def load_chunks(self, test_data_dir: Path) -> List[Dict[str, Any]]:
        """Load all documents from test_data directory and return chunks."""
        pass
    
    @abstractmethod
    def load_all_documents(self, test_data_dir: Path) -> List[Dict[str, Any]]:
        """Load all documents with format-specific chunking."""
        pass