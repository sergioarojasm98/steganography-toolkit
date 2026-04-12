"""
Esteganografía basada en LSB (Least Significant Bit).

Este módulo implementa la técnica más simple de esteganografía: modificar
el bit menos significativo de cada canal de color para ocultar datos.

Es la técnica más eficiente en términos de capacidad, pero la menos
resistente a modificaciones de la imagen.

Uso:
    python lsb.py --input <carpeta_entrada> --output <carpeta_salida> --text <archivo_texto>
    python lsb.py --input imagen.png --output stego.png --message "Texto secreto"
"""

import argparse
import os
import random
import sys
import traceback

from PIL import Image

# Agregar el directorio padre al path para importar common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment


def text_to_bits(text: str) -> str:
    """
    Convierte una cadena de texto en una cadena de bits.

    Args:
        text: Texto a convertir.

    Returns:
        Cadena binaria (8 bits por carácter).
    """
    return "".join(format(ord(char), "08b") for char in text)


def max_text_length(image_path: str) -> int:
    """
    Determina la longitud máxima de caracteres que se pueden ocultar.

    Args:
        image_path: Ruta a la imagen.

    Returns:
        Número máximo de caracteres.
    """
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    max_bits = width * height * 3  # 3 bits por píxel (RGB)
    max_chars = max_bits // 8 - 1  # 8 bits por carácter menos el byte nulo
    return max_chars


def hide_text_in_image(image_path: str, text: str, output_path: str) -> bool:
    """
    Oculta una cadena de texto en una imagen usando LSB.

    Args:
        image_path: Ruta a la imagen original.
        text: Texto a ocultar.
        output_path: Ruta donde guardar la imagen esteganográfica.

    Returns:
        True si el texto se ocultó completamente, False en caso contrario.
    """
    image = Image.open(image_path).convert("RGBA")
    width, height = image.size
    pixels = image.load()

    # Agregar byte nulo como delimitador
    text += "\0"
    text_bits = text_to_bits(text)

    bit_index = 0
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            if bit_index < len(text_bits):
                r = (r & ~1) | int(text_bits[bit_index])
                bit_index += 1
            if bit_index < len(text_bits):
                g = (g & ~1) | int(text_bits[bit_index])
                bit_index += 1
            if bit_index < len(text_bits):
                b = (b & ~1) | int(text_bits[bit_index])
                bit_index += 1

            pixels[x, y] = (r, g, b, a)

    image.save(output_path)
    return bit_index == len(text_bits)


def read_text_file(file_path: str) -> str:
    """Lee el contenido de un archivo de texto."""
    with open(file_path, encoding="utf-8") as file:
        return file.read()


def process_folder(input_folder: str, output_folder: str, text_file: str):
    """
    Process every PNG in *input_folder*, embedding random fragments of *text_file*.

    Args:
        input_folder: Folder containing cover images.
        output_folder: Destination folder for the stego images.
        text_file: Path to a UTF-8 text file used as the source of payloads.

    Telegram notifications are sent on start, completion, and per-image errors
    when ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` env vars are set;
    otherwise they are silently skipped.
    """
    send_telegram_message(f"LSB: Iniciando procesamiento de {input_folder}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    text_data = read_text_file(text_file)

    for img_file in os.listdir(input_folder):
        if img_file.lower().endswith(".png"):
            img_path = os.path.join(input_folder, img_file)

            try:
                max_chars = max_text_length(img_path)
                step = max_chars // 5
                text_lengths = [i * step for i in range(1, 6)]

                for text_len in text_lengths:
                    random_start = random.randint(0, len(text_data) - text_len)
                    random_text = text_data[random_start : random_start + text_len]
                    output_img_name = f"{os.path.splitext(img_file)[0]}_LSB_{text_len}{os.path.splitext(img_file)[1]}"
                    output_path = os.path.join(output_folder, output_img_name)
                    success = hide_text_in_image(img_path, random_text, output_path)

                    if success:
                        print(f"Procesado: {output_img_name}")
                    else:
                        error_msg = f"Error: texto muy largo para {img_file}"
                        print(error_msg)
                        send_telegram_message(error_msg)

            except Exception:
                error_msg = f"Error al procesar {img_file}: {traceback.format_exc()}"
                print(error_msg)
                send_telegram_message(error_msg)

    send_telegram_message("LSB: Procesamiento completado")


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Esteganografía LSB - Ocultar texto en imágenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Procesar una sola imagen
  python lsb.py --input imagen.png --output stego.png --message "Texto secreto"

  # Procesar una carpeta completa
  python lsb.py --input ./imagenes/ --output ./salida/ --text datos.txt
          """,
    )

    parser.add_argument("--input", "-i", required=True, help="Imagen o carpeta de entrada")
    parser.add_argument("--output", "-o", required=True, help="Imagen o carpeta de salida")
    parser.add_argument("--message", "-m", help="Mensaje a ocultar (para imagen individual)")
    parser.add_argument(
        "--text", "-t", help="Archivo de texto a ocultar (para procesamiento por lotes)"
    )

    return parser.parse_args()


def main():
    """Función principal."""
    setup_environment()
    args = parse_args()

    send_telegram_message("LSB: Programa iniciado")

    try:
        if os.path.isfile(args.input):
            # Procesar una sola imagen
            if not args.message:
                print("Error: Se requiere --message para procesar una imagen individual")
                sys.exit(1)
            success = hide_text_in_image(args.input, args.message, args.output)
            if success:
                print(f"Imagen procesada: {args.output}")
            else:
                print("Error: El mensaje es muy largo para esta imagen")
                sys.exit(1)
        else:
            # Procesar carpeta
            if not args.text:
                print("Error: Se requiere --text para procesar una carpeta")
                sys.exit(1)
            process_folder(args.input, args.output, args.text)
    except Exception:
        error_msg = f"Error: {traceback.format_exc()}"
        print(error_msg)
        send_telegram_message(error_msg)
        sys.exit(1)

    send_telegram_message("LSB: Programa finalizado")


if __name__ == "__main__":
    main()
