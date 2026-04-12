"""
Decodificador de esteganografía LSB.

Extrae texto oculto de imágenes procesadas con lsb.py.

Uso:
    python decode_lsb.py --input imagen_stego.png
"""

import argparse
import os
import sys
import traceback

from PIL import Image

# Agregar el directorio padre al path para importar common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment


def bits_to_text(bits: str) -> str:
    """
    Convierte una cadena de bits en texto ASCII.

    Args:
        bits: Cadena de '0' y '1'.

    Returns:
        Texto decodificado.
    """
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8 : (b + 1) * 8]
        chars.append(chr(int(byte, 2)))
    return "".join(chars)


def decode_text_from_image(image_path: str) -> str:
    """
    Extrae texto oculto de una imagen usando LSB.

    El texto fue codificado con un byte nulo '\0' al final como delimitador.
    Los bits están almacenados secuencialmente en R, G, B de cada píxel.

    Args:
        image_path: Ruta a la imagen esteganográfica.

    Returns:
        Texto extraído.
    """
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    pixels = image.load()
    bits = []

    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            bits.append(r & 1)
            bits.append(g & 1)
            bits.append(b & 1)

    bits_as_string = "".join(str(bit) for bit in bits)

    # Buscar el byte nulo (delimitador) en posiciones múltiplos de 8 (alineado a byte)
    end_of_text = -1
    for i in range(0, len(bits_as_string) - 7, 8):
        if bits_as_string[i : i + 8] == "00000000":
            end_of_text = i
            break

    # Si no se encuentra el delimitador, usar todos los bits disponibles
    if end_of_text == -1:
        end_of_text = len(bits_as_string)

    bits_as_string = bits_as_string[:end_of_text]

    return bits_to_text(bits_as_string)


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Decodificador LSB - Extraer texto oculto de imágenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python decode_lsb.py --input stego.png
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Stego image to decode")

    return parser.parse_args()


def main():
    """CLI entry point."""
    setup_environment()
    args = parse_args()

    send_telegram_message("LSB decode: program started")

    try:
        hidden_message = decode_text_from_image(args.input)
        print(f"Hidden message ({len(hidden_message)} chars):")
        print(hidden_message)
        preview = hidden_message[:50] + "..." if len(hidden_message) > 50 else hidden_message
        send_telegram_message(f"LSB decode: {preview} ({len(hidden_message)} chars)")
    except Exception:
        error_msg = f"Error: {traceback.format_exc()}"
        print(error_msg)
        send_telegram_message(error_msg)
        sys.exit(1)

    send_telegram_message("LSB decode: program finished")


if __name__ == "__main__":
    main()
