"""
Herramienta para eliminar perfiles de color sRGB de imágenes.

Algunos procesos de esteganografía pueden verse afectados por los
perfiles de color incrustados. Esta herramienta los elimina.

Uso:
    python remove_sRGB.py --input <carpeta>
"""

import argparse
import os
import sys

from PIL import Image

# Agregar el directorio padre al path para importar common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment


def remove_srgb_profile(image_path: str) -> bool:
    """
    Elimina el perfil de color sRGB de una imagen.

    Args:
        image_path: Ruta a la imagen.

    Returns:
        True si se eliminó el perfil, False si no tenía.
    """
    try:
        image = Image.open(image_path)

        if "icc_profile" in image.info:
            del image.info["icc_profile"]
            image.save(image_path, icc_profile=None)
            image.close()
            return True

        image.close()
        return False

    except Exception as e:
        print(f"Error al procesar {image_path}: {e}")
        return False


def process_folder(folder_path: str) -> tuple:
    """
    Procesa todas las imágenes PNG de una carpeta.

    Args:
        folder_path: Carpeta con imágenes.

    Returns:
        Tupla (procesadas, modificadas).
    """
    processed = 0
    modified = 0

    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(".png"):
            image_path = os.path.join(folder_path, img_file)
            processed += 1

            if remove_srgb_profile(image_path):
                modified += 1
                print(f"Perfil sRGB eliminado: {img_file}")

    return processed, modified


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Eliminar perfiles de color sRGB de imágenes",
    )

    parser.add_argument("--input", "-i", required=True, help="Carpeta con imágenes")
    parser.add_argument(
        "--no-notify", action="store_true", help="Desactivar notificaciones de Telegram"
    )

    return parser.parse_args()


def main():
    """Función principal."""
    setup_environment()
    args = parse_args()

    notify = not args.no_notify

    processed, modified = process_folder(args.input)

    result = f"Procesadas: {processed}, Modificadas: {modified}"
    print(result)

    if notify:
        send_telegram_message(f"remove_sRGB: {result}")


if __name__ == "__main__":
    main()
