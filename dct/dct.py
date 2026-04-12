"""DCT-based image steganography using Quantization Index Modulation (QIM)
with intra-channel repetition coding.

This module hides text by applying the 2D Discrete Cosine Transform to 8x8
pixel blocks of each RGB channel independently, then quantizing the low-
frequency AC coefficients via QIM. Each bit is written to ``REPETITIONS``
distinct coefficient positions per channel, and decoding recovers the bit
by majority voting first within each channel and then across the three
RGB channels (so each bit is sampled ``3 × REPETITIONS`` times).

The previous implementation modified the LSB of JPEG-quantized coefficients
in the YCrCb luminance channel — this was not round-trip stable because the
``BGR → YCrCb → uint8 → BGR → YCrCb`` double color-space conversion
introduced ±3–4 coefficient-level perturbation, flipping LSBs on ~80 % of
real photographs. The new implementation works directly in the BGR pixel
domain (no color-space conversion) and uses QIM instead of LSB, achieving
~80 % success on real photographs, with a self-verify guard that ensures
corrupted outputs never persist to disk.

Usage:
    python dct.py --input image.png --output stego.png --message "Secret"
    python dct.py --input ./images/ --output ./stego/ --text data.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import traceback

import cv2
import numpy as np

# Add the parent directory to the path so we can import the common package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment

# QIM quantization step. The IDCT → uint8 → DCT round-trip perturbs raw
# coefficients by ±1–3; Q=16 (tolerance ±4) gives safe margin.
QUANTIZATION_STEP = 16
HALF_STEP = QUANTIZATION_STEP / 2.0

# Each bit is replicated to REPETITIONS distinct positions per channel.
REPETITIONS = 5

# Only the first N_AC_POSITIONS AC coefficients (in zigzag order) of each
# 8x8 block are used. Low-frequency coefficients (early zigzag positions)
# have larger magnitude and survive the round-trip better than high-frequency
# ones. 20 is a good balance between capacity and reliability.
N_AC_POSITIONS = 20


class DctVerifyError(RuntimeError):
    """Raised when the post-encode self-verification round-trip fails."""


# ---------------------------------------------------------------------------
# QIM core
# ---------------------------------------------------------------------------


def qim_embed(coefficient: float, bit: int) -> float:
    """Quantize *coefficient* to the lattice that encodes *bit*."""
    if bit == 0:
        return round(coefficient / QUANTIZATION_STEP) * QUANTIZATION_STEP
    return round((coefficient - HALF_STEP) / QUANTIZATION_STEP) * QUANTIZATION_STEP + HALF_STEP


def qim_extract(coefficient: float) -> int:
    """Recover the bit encoded in *coefficient* by QIM."""
    return int(round(coefficient / HALF_STEP)) & 1


# ---------------------------------------------------------------------------
# 8x8 block utilities
# ---------------------------------------------------------------------------


def _blocks_from_channel(channel: np.ndarray) -> list[np.ndarray]:
    """Split a single-channel image into 8x8 DCT blocks."""
    h, w = channel.shape
    blocks: list[np.ndarray] = []
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            blocks.append(cv2.dct(np.float64(channel[y : y + 8, x : x + 8])))
    return blocks


def _channel_from_blocks(blocks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    """Reconstruct a channel from 8x8 DCT blocks via IDCT."""
    out = np.zeros((h, w), dtype=np.float64)
    idx = 0
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            out[y : y + 8, x : x + 8] = cv2.idct(blocks[idx])
            idx += 1
    return out


# ---------------------------------------------------------------------------
# Slot arithmetic (shared between embed and extract)
# ---------------------------------------------------------------------------


def _flatten_ac(blocks: list[np.ndarray]) -> np.ndarray:
    """Concatenate the first N_AC_POSITIONS AC coefficients from all blocks."""
    return np.concatenate([b.flatten()[1 : 1 + N_AC_POSITIONS] for b in blocks]).astype(np.float64)


def _unflatten_ac(blocks: list[np.ndarray], flat_ac: np.ndarray) -> list[np.ndarray]:
    """Write the modified AC vector back into per-block arrays."""
    result: list[np.ndarray] = []
    offset = 0
    for b in blocks:
        new_b = b.flatten().copy()
        new_b[1 : 1 + N_AC_POSITIONS] = flat_ac[offset : offset + N_AC_POSITIONS]
        offset += N_AC_POSITIONS
        result.append(new_b.reshape(8, 8))
    return result


def _slot_positions(n_bits: int, total_slots: int) -> list[int]:
    """Spread *n_bits* × REPETITIONS positions uniformly across *total_slots*."""
    if n_bits == 0:
        return []
    needed = n_bits * REPETITIONS
    if needed > total_slots:
        raise ValueError(
            f"Message of {n_bits} bits with {REPETITIONS}x replication "
            f"needs {needed} AC slots, only {total_slots} available"
        )
    step = max(1, total_slots // needed)
    return [
        min((i * REPETITIONS + r) * step, total_slots - 1)
        for i in range(n_bits)
        for r in range(REPETITIONS)
    ]


# ---------------------------------------------------------------------------
# Binary encoding helpers
# ---------------------------------------------------------------------------


def _text_to_bits(text: str) -> str:
    """ASCII text → binary string (no header — length is passed out-of-band)."""
    return "".join(format(b, "08b") for b in text.encode("ascii"))


def _bits_to_text(bits: str) -> str:
    """Binary string → ASCII text."""
    out = bytearray()
    for i in range(0, len(bits) - 7, 8):
        out.append(int(bits[i : i + 8], 2))
    return out.decode("ascii")


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


def max_text_length(image_path: str) -> int:
    """Return the maximum number of ASCII chars that can be hidden."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    n_blocks = (h // 8) * (w // 8)
    total_ac = n_blocks * N_AC_POSITIONS
    return max(0, (total_ac // REPETITIONS) // 8)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def hide_text_in_image(
    image_path: str,
    text: str,
    output_path: str,
    *,
    verify: bool = True,
) -> None:
    """Hide *text* in *image_path* and save the stego image to *output_path*.

    Uses QIM + 5x repetition on the first 20 AC coefficients of each 8x8
    DCT block, independently in all three BGR channels. 2-of-3 channel
    majority vote is applied at decode time.

    If ``verify`` is True (default) and *text* is non-empty, the function
    decodes the just-written stego and raises ``DctVerifyError`` if the
    round-trip does not recover *text* byte-for-byte.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    message_bits = _text_to_bits(text)
    h, w = img.shape[:2]
    stego_chans: list[np.ndarray] = []

    for chan in cv2.split(img):
        blocks = _blocks_from_channel(chan)
        flat_ac = _flatten_ac(blocks)

        if len(message_bits) > 0:
            positions = _slot_positions(len(message_bits), len(flat_ac))
            for i, bit_char in enumerate(message_bits):
                bit = int(bit_char)
                for r in range(REPETITIONS):
                    flat_ac[positions[i * REPETITIONS + r]] = qim_embed(
                        flat_ac[positions[i * REPETITIONS + r]], bit
                    )

        new_blocks = _unflatten_ac(blocks, flat_ac)
        recon = _channel_from_blocks(new_blocks, h, w)
        stego_chans.append(np.clip(recon, 0, 255).astype(np.uint8))

    cv2.imwrite(output_path, cv2.merge(stego_chans))

    if verify and text:
        try:
            recovered = decode_from_file(output_path, text_length=len(text))
        except Exception as e:
            _safe_remove(output_path)
            raise DctVerifyError(f"Verify: decode raised {type(e).__name__}: {e}") from e
        if recovered != text:
            _safe_remove(output_path)
            raise DctVerifyError(
                "Verify: round-trip recovered text does not match input. "
                "This image is in the small fraction where DCT-QIM cannot "
                "guarantee bit-exact recovery; pick a different cover."
            )


def decode_from_file(stego_path: str, text_length: int | None = None) -> str:
    """Decode the hidden ASCII text from a stego image.

    *text_length* is the number of characters to extract. If ``None``, the
    length is parsed from the filename (expected pattern ``*_DCT_<n>.ext``).
    """
    if text_length is None:
        text_length = _get_length_from_filename(stego_path)

    img = cv2.imread(stego_path)
    if img is None:
        raise ValueError(f"Could not read image: {stego_path}")

    n_bits = text_length * 8
    h, w = img.shape[:2]
    n_blocks = (h // 8) * (w // 8)
    total_ac = n_blocks * N_AC_POSITIONS
    all_bits = _extract_bits_multichannel(img, n_bits, total_ac)
    return _bits_to_text(all_bits)


def _get_length_from_filename(path: str) -> int:
    """Parse ``*_DCT_<n>.ext`` → n."""
    basename = os.path.splitext(os.path.basename(path))[0]
    if "_DCT_" not in basename:
        raise ValueError(f"Cannot parse length from {path!r}: expected *_DCT_<length>.ext")
    return int(basename.split("_DCT_")[-1])


def _extract_bits_multichannel(img: np.ndarray, n_bits: int, total_ac: int) -> str:
    """Extract *n_bits* from all 3 channels with 2-of-3 majority vote."""
    bits_per_chan: list[list[int]] = []
    for chan in cv2.split(img):
        blocks = _blocks_from_channel(chan)
        flat_ac = _flatten_ac(blocks)
        positions = _slot_positions(n_bits, total_ac)
        per_bit: list[int] = []
        for i in range(n_bits):
            votes = [
                qim_extract(flat_ac[positions[i * REPETITIONS + r]]) for r in range(REPETITIONS)
            ]
            per_bit.append(1 if sum(votes) > REPETITIONS // 2 else 0)
        bits_per_chan.append(per_bit)

    return "".join(
        "1" if (bits_per_chan[0][i] + bits_per_chan[1][i] + bits_per_chan[2][i]) >= 2 else "0"
        for i in range(n_bits)
    )


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def read_text_file(file_path: str) -> str:
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def process_folder(input_folder: str, output_folder: str, text_file: str) -> None:
    """Process every PNG in *input_folder*, embedding random fragments of *text_file*.

    Images that fail self-verification are reported and skipped.
    """
    send_telegram_message(f"DCT: started processing {input_folder}")
    os.makedirs(output_folder, exist_ok=True)
    text_data = read_text_file(text_file)

    n_ok = n_skip = n_err = 0
    for img_file in sorted(os.listdir(input_folder)):
        if not img_file.lower().endswith(".png"):
            continue
        img_path = os.path.join(input_folder, img_file)
        try:
            cap = max_text_length(img_path)
            step = max(1, cap // 5)
            for mul in range(1, 6):
                text_len = mul * step
                if text_len > len(text_data):
                    continue
                start = random.randint(0, len(text_data) - text_len)
                msg = text_data[start : start + text_len]
                stem, ext = os.path.splitext(img_file)
                out_name = f"{stem}_DCT_{text_len}{ext}"
                out_path = os.path.join(output_folder, out_name)
                try:
                    hide_text_in_image(img_path, msg, out_path)
                    print(f"Processed: {out_name}")
                    n_ok += 1
                except DctVerifyError:
                    print(f"Skipped (verify failed): {out_name}")
                    n_skip += 1
        except Exception:
            print(f"Error: {img_file}: {traceback.format_exc()}")
            send_telegram_message(f"Error: {img_file}: {traceback.format_exc()}")
            n_err += 1

    summary = f"DCT: complete — ok={n_ok}, skipped={n_skip}, errors={n_err}"
    print(summary)
    send_telegram_message(summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DCT steganography — hide text using QIM on 8x8 DCT blocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dct.py --input image.png --output stego.png --message "Secret"
  python dct.py --input ./images/ --output ./output/ --text data.txt
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Input image or folder")
    parser.add_argument("--output", "-o", required=True, help="Output image or folder")
    parser.add_argument("--message", "-m", help="Message to hide")
    parser.add_argument("--text", "-t", help="Text file for folder mode")
    parser.add_argument("--no-verify", action="store_true", help="Skip verify (faster but unsafe)")
    return parser.parse_args()


def main() -> None:
    setup_environment()
    args = parse_args()
    send_telegram_message("DCT: program started")
    try:
        if os.path.isfile(args.input):
            if not args.message:
                print("Error: --message required for single-image mode")
                sys.exit(1)
            hide_text_in_image(args.input, args.message, args.output, verify=not args.no_verify)
            print(f"Wrote: {args.output}")
        else:
            if not args.text:
                print("Error: --text required for folder mode")
                sys.exit(1)
            process_folder(args.input, args.output, args.text)
    except Exception:
        print(f"Error: {traceback.format_exc()}")
        send_telegram_message(f"Error: {traceback.format_exc()}")
        sys.exit(1)
    send_telegram_message("DCT: program finished")


if __name__ == "__main__":
    main()
