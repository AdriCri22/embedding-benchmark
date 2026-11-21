"""
Config module for the embedding benchmark suite.
"""
from .interface import ConfigInterface

_config: ConfigInterface = None


def get_config() -> ConfigInterface:
    """Get the config implementation singleton."""
    global _config
    if _config is None:
        from .internal.config_impl import ConfigImpl
        _config = ConfigImpl()
    return _config


__all__ = ["ConfigInterface", "get_config"]