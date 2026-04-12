"""
Herramienta para redimensionar imágenes.

Redimensiona imágenes a una resolución específica. Las imágenes más pequeñas
que la resolución objetivo son eliminadas.

Uso:
    python resize_images.py --input <carpeta> --resolution 480x360
"""

import argparse
import os
import sys

from PIL import Image

# Agregar el directorio padre al path para importar common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment


def resize_images(directory: str, resolution: tuple) -> tuple:
    """
    Redimensiona imágenes a la resolución especificada.

    Las imágenes más pequeñas que la resolución objetivo son eliminadas.

    Args:
        directory: Carpeta con imágenes.
        resolution: Tupla (ancho, alto).

    Returns:
        Tupla (modificadas, eliminadas).
    """
    modified_count = 0
    deleted_count = 0

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(directory, filename)

            try:
                img = Image.open(img_path)

                if img.size != resolution:
                    # Si la imagen es más pequeña, eliminarla
                    if img.size[0] < resolution[0] or img.size[1] < resolution[1]:
                        img.close()
                        os.remove(img_path)
                        deleted_count += 1
                        continue

                    # Redimensionar manteniendo aspecto
                    img.thumbnail((resolution[0], resolution[1]))

                    # Recortar al centro
                    left_margin = (img.width - resolution[0]) / 2
                    top_margin = (img.height - resolution[1]) / 2
                    img = img.crop(
                        (
                            left_margin,
                            top_margin,
                            img.width - left_margin,
                            img.height - top_margin,
                        )
                    )

                    img.save(img_path)
                    modified_count += 1

                img.close()

            except Exception as e:
                print(f"Error al procesar {filename}: {e}")

    return modified_count, deleted_count


def parse_resolution(resolution_str: str) -> tuple:
    """Convierte string '480x360' a tupla (480, 360)."""
    parts = resolution_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Formato de resolución inválido: {resolution_str}. Usar: 480x360")
    return (int(parts[0]), int(parts[1]))


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Redimensionar imágenes a una resolución específica",
    )

    parser.add_argument("--input", "-i", required=True, help="Carpeta con imágenes")
    parser.add_argument(
        "--resolution", "-r", default="480x360", help="Resolución objetivo (ej: 480x360)"
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

    modified, deleted = resize_images(args.input, resolution)

    result = f"Modificadas: {modified}, Eliminadas: {deleted}"
    print(result)

    if notify:
        send_telegram_message(f"resize_images: {result}")


if __name__ == "__main__":
    main()
