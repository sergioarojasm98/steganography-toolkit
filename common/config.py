"""
Módulo de configuración.

Carga configuración desde variables de entorno o archivo .env
"""

import os
from pathlib import Path


def get_config() -> dict:
    """
    Obtiene la configuración del proyecto.

    Intenta cargar desde variables de entorno.
    Si existe python-dotenv, también carga desde .env

    Returns:
        Diccionario con la configuración.
    """
    # Intentar cargar .env si python-dotenv está disponible
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass  # python-dotenv no instalado, usar solo variables de entorno

    return {
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
    }


def setup_environment():
    """
    Configura el entorno cargando variables desde .env si existe.
    Llamar al inicio del programa.
    """
    get_config()  # Esto carga el .env si existe
