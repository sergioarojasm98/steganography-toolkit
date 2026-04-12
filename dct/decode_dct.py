"""Decoder for DCT-QIM steganography (companion to ``dct.dct``).

Reads QIM-encoded bits from the low-frequency AC coefficients of the 8x8
DCT blocks in all three BGR channels, applies majority vote, and recovers
the original ASCII text. The message length is parsed from the filename
(pattern ``*_DCT_<length>.ext``) or passed explicitly via ``--length``.

Usage:
    python decode_dct.py --input stego_DCT_100.png
    python decode_dct.py --input stego.png --length 100
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment
from dct.dct import decode_from_file


def decode_text_from_image(stego_image_path: str, text_length: int | None = None) -> str:
    """Decode the hidden ASCII text from a DCT-QIM stego image."""
    return decode_from_file(stego_image_path, text_length=text_length)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DCT-QIM decoder — extract hidden text from a stego image",
        epilog="Example: python decode_dct.py --input stego_DCT_100.png",
    )
    parser.add_argument("--input", "-i", required=True, help="Stego image")
    parser.add_argument("--length", "-l", type=int, help="Payload length in chars")
    return parser.parse_args()


def main() -> None:
    setup_environment()
    args = parse_args()
    send_telegram_message("DCT decode: program started")
    try:
        text = decode_text_from_image(args.input, text_length=args.length)
        print(f"Hidden message ({len(text)} chars):")
        print(text)
        send_telegram_message(f"DCT decode: {text[:50]}... ({len(text)} chars)")
    except Exception:
        print(f"Error: {traceback.format_exc()}")
        send_telegram_message(f"Error: {traceback.format_exc()}")
        sys.exit(1)
    send_telegram_message("DCT decode: program finished")


if __name__ == "__main__":
    main()
