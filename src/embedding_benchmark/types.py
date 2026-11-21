"""
Shared types and enums for the embedding benchmark suite.
"""
from enum import Enum


class BenchmarkLang(str, Enum):
    CA = "ca"
    ES = "es"
    EN = "en"


class BenchmarkMode(str, Enum):
    FULL = "full"
    QUICK = "quick"
    FORMATS = "formats"
    MINI = "mini"


class SetupTarget(str, Enum):
    MAC = "mac"
    WIN = "win"
    LINUX = "linux"
    AUTO = "auto"