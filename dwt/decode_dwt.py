"""Decoder for DWT-QIM steganography with intra-channel repetition coding.

The QIM lattice used by the encoder is reversible by construction:
each cD coefficient is rounded to the nearest half-step of ``Q``, then
the parity of that integer index gives back the embedded bit. Bits are
read from the same coefficient positions written by the encoder
(``REPETITIONS`` replicas per bit, spread uniformly across the cD
vector), combined first by majority vote within each channel and then
by 2-of-3 majority vote across the three RGB channels.

Usage:
    python decode_dwt.py --input stego.png --length 100
    python decode_dwt.py --input stego_DWT_100.png   # length parsed from filename
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import cv2
import numpy as np
import pywt

# Add the parent directory to the path so we can import the common package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment

# Must match dwt.py
QUANTIZATION_STEP = 24
HALF_STEP = QUANTIZATION_STEP / 2.0
REPETITIONS = 5


def qim_extract(coefficient: float) -> int:
    """Recover the bit encoded in *coefficient* by the QIM encoder."""
    return int(round(coefficient / HALF_STEP)) & 1


def _slot_positions(n_bits: int, total_slots: int) -> list[int]:
    """Compute the cD positions used by the encoder (must mirror dwt.dwt._slot_positions)."""
    if n_bits == 0:
        return []
    needed = n_bits * REPETITIONS
    if needed > total_slots:
        raise ValueError(
            f"Cannot decode {n_bits} bits with {REPETITIONS}x replication "
            f"from a channel with only {total_slots} cD slots"
        )
    step = total_slots // needed
    return [(i * REPETITIONS + r) * step for i in range(n_bits) for r in range(REPETITIONS)]


def extract_from_channel(channel: np.ndarray, n_bits: int) -> list[int]:
    """Extract *n_bits* bits from *channel*, voting across REPETITIONS replicas."""
    _, (_, _, cD) = pywt.dwt2(channel.astype(np.float64), "haar")
    flat = cD.flatten()
    positions = _slot_positions(n_bits, flat.size)
    bits: list[int] = []
    for i in range(n_bits):
        votes = [qim_extract(flat[positions[i * REPETITIONS + r]]) for r in range(REPETITIONS)]
        bits.append(1 if sum(votes) > REPETITIONS // 2 else 0)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    """Pack a flat list of bits into raw bytes (truncates a trailing < 8 bits)."""
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)


def get_text_length_from_filename(filename: str) -> int:
    """Parse the message length out of a stego filename of the form ``*_DWT_<n>.ext``."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    if "_DWT_" not in basename:
        raise ValueError(
            f"Cannot parse length from filename {filename!r}: expected pattern *_DWT_<length>.ext"
        )
    return int(basename.split("_DWT_")[-1])


def decode_text_from_image(stego_image_path: str, text_length: int | None = None) -> bytes:
    """Decode the hidden payload from *stego_image_path*.

    Returns the recovered bytes verbatim. The caller is responsible for
    decoding bytes to a string (e.g. ``.decode('utf-8', errors='replace')``).

    If *text_length* is None, the length is parsed from the filename.
    """
    if text_length is None:
        text_length = get_text_length_from_filename(stego_image_path)

    n_bits = text_length * 8
    img = cv2.imread(stego_image_path)
    if img is None:
        raise ValueError(f"Could not read image: {stego_image_path}")

    b, g, r = cv2.split(img)
    bits_b = extract_from_channel(b, n_bits)
    bits_g = extract_from_channel(g, n_bits)
    bits_r = extract_from_channel(r, n_bits)

    final_bits = [1 if (bits_b[i] + bits_g[i] + bits_r[i]) >= 2 else 0 for i in range(n_bits)]
    return bits_to_bytes(final_bits)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DWT-QIM decoder — extract hidden text from a stego image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python decode_dwt.py --input image_DWT_100.png
  python decode_dwt.py --input image.png --length 100
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Stego image to decode")
    parser.add_argument("--length", "-l", type=int, help="Payload length in bytes")
    return parser.parse_args()


def main() -> None:
    setup_environment()
    args = parse_args()
    send_telegram_message("DWT decode: program started")

    try:
        payload = decode_text_from_image(args.input, args.length)
        try:
            text = payload.decode("utf-8")
            print(f"Hidden message ({len(text)} chars): {text}")
        except UnicodeDecodeError:
            print(f"Hidden payload ({len(payload)} bytes, not valid UTF-8): {payload!r}")
        send_telegram_message(f"DWT decode: recovered {len(payload)} bytes")
    except Exception:
        error_msg = f"Error: {traceback.format_exc()}"
        print(error_msg)
        send_telegram_message(error_msg)
        sys.exit(1)

    send_telegram_message("DWT decode: program finished")


if __name__ == "__main__":
    main()
