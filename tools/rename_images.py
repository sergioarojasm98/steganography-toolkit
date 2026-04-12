"""
Herramienta para renombrar imágenes secuencialmente.

Renombra todas las imágenes con formato IMAGE000001.png, IMAGE000002.png, etc.

Uso:
    python rename_images.py --input <carpeta>
"""

import argparse
import os
import sys

# Agregar el directorio padre al path para importar common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment


def rename_files(folder_path: str, prefix: str = "IMAGE", batch_size: int = 10000) -> int:
    """
    Renombra archivos de imagen secuencialmente.

    Args:
        folder_path: Carpeta con imágenes.
        prefix: Prefijo para los nombres (default: IMAGE).
        batch_size: Tamaño del lote (para mantener numeración continua).

    Returns:
        Número de archivos renombrados.
    """
    file_list = os.listdir(folder_path)
    renamed_count = 0

    for i, file_name in enumerate(file_list):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            extension = file_name.split(".")[-1]

            # Calcular número secuencial
            batch_number = (i // batch_size) + 1
            file_number = i % batch_size
            sequence = (batch_number - 1) * batch_size + file_number + 1

            new_file_name = f"{prefix}{sequence:06d}.{extension}"

            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_file_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                renamed_count += 1

    return renamed_count


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Renombrar imágenes secuencialmente",
    )

    parser.add_argument("--input", "-i", required=True, help="Carpeta con imágenes")
    parser.add_argument(
        "--prefix", "-p", default="IMAGE", help="Prefijo para nombres (default: IMAGE)"
    )
    parser.add_argument(
        "--no-notify", action="store_true", help="Desactivar notificaciones de Telegram"
    )

    return parser.parse_args()


def main():
    """Función principal."""
    setup_environment()
    args = parse_args()

    notify = not args.no_notify

    count = rename_files(args.input, args.prefix)

    result = f"Archivos renombrados: {count}"
    print(result)

    if notify:
        send_telegram_message(f"rename_images: {result}")


if __name__ == "__main__":
    main()
