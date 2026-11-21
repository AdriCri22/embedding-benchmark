"""
Database module for the embedding benchmark suite.
"""
from .interface import DatabaseInterface

_database: DatabaseInterface = None


def get_database() -> DatabaseInterface:
    """Get the database implementation singleton."""
    global _database
    if _database is None:
        from .internal.database_impl import DatabaseImpl
        _database = DatabaseImpl()
    return _database


__all__ = ["DatabaseInterface", "get_database"]