"""
Herramienta para verificar imágenes en una carpeta.

Cuenta cuántas imágenes tienen una resolución específica.

Uso:
    python check_images.py --input <carpeta> --resolution 480x360
"""

import argparse
import os
import sys

from PIL import Image

# Agregar el directorio padre al path para importar common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment


def count_images_in_resolution(directory: str, resolution: tuple) -> int:
    """
    Cuenta imágenes con una resolución específica.

    Args:
        directory: Carpeta con imágenes.
        resolution: Tupla (ancho, alto).

    Returns:
        Número de imágenes con la resolución especificada.
    """
    count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(directory, filename)
            try:
                img = Image.open(img_path)
                if img.size == resolution:
                    count += 1
                img.close()
            except Exception as e:
                print(f"Error al abrir {filename}: {e}")
    return count


def parse_resolution(resolution_str: str) -> tuple:
    """Convierte string '480x360' a tupla (480, 360)."""
    parts = resolution_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Formato de resolución inválido: {resolution_str}. Usar: 480x360")
    return (int(parts[0]), int(parts[1]))


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Verificar imágenes - Contar imágenes con resolución específica",
    )

    parser.add_argument("--input", "-i", required=True, help="Carpeta con imágenes")
    parser.add_argument(
        "--resolution", "-r", default="480x360", help="Resolución a verificar (ej: 480x360)"
    )
    parser.add_argument(
        "--no-notify", action="store_true", help="Desactivar notificaciones de Telegram"
    )

    return parser.parse_args()


def main():
    """Función principal."""
    setup_environment()
    args = parse_args()

    resolution = parse_resolution(args.resolution)
    notify = not args.no_notify

    count = count_images_in_resolution(args.input, resolution)

    total = len(
        [f for f in os.listdir(args.input) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    result = f"Imágenes con resolución {args.resolution}: {count}/{total}"
    print(result)

    if notify:
        send_telegram_message(f"check_images: {result}")


if __name__ == "__main__":
    main()
