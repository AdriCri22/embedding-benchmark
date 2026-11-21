"""
Data loader module for the embedding benchmark suite.
"""
from .interface import DataLoaderInterface

_data_loader: DataLoaderInterface = None


def get_data_loader() -> DataLoaderInterface:
    """Get the data loader implementation singleton."""
    global _data_loader
    if _data_loader is None:
        from .internal.data_loader_impl import DataLoaderImpl
        _data_loader = DataLoaderImpl()
    return _data_loader


__all__ = ["DataLoaderInterface", "get_data_loader"]