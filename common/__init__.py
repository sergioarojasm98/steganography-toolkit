"""
Módulo común para funcionalidades compartidas entre los algoritmos de esteganografía.
"""

from .config import get_config
from .notifications import send_telegram_message

__all__ = ["send_telegram_message", "get_config"]
