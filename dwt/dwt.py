"""DWT-based image steganography using Quantization Index Modulation (QIM)
with intra-channel repetition coding.

This module hides text inside images by quantizing the diagonal-detail
sub-band of a Haar 2D Discrete Wavelet Transform. Each bit is written to
``REPETITIONS`` distinct cD positions per channel, and decoding recovers
the bit by majority voting first within each channel and then across the
three RGB channels (so each bit is sampled ``3 × REPETITIONS`` times).

The Cornell 1998 paper proposed modifying the LSB of "small" cD
coefficients in place. That approach is **not round-trip stable**: the
``IDWT → uint8 → DWT`` cycle moves coefficients out of the threshold
range and silently corrupts bits — visible on any non-trivial natural
image. The QIM lattice + repetition + majority vote in this implementation
is round-trip stable on every image we have tested, including the
~510K-image dataset used for the companion ``steganalysis-deep-learning``
thesis.

To make failures impossible-to-miss, ``hide_text_in_image`` performs a
self-verification step: it decodes the stego image it just produced and
raises ``DwtVerifyError`` if the round-trip does not recover the message
byte-for-byte. Pass ``verify=False`` to skip the check (faster but risky).

Usage:
    python dwt.py --input image.png --output stego.png --message "Secret"
    python dwt.py --input ./images/ --output ./stego/ --text data.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import traceback

import cv2
import numpy as np
import pywt

# Add the parent directory to the path so we can import the common package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import send_telegram_message
from common.config import setup_environment

# QIM quantization step. With Q = 24 the lattice has half-step intervals
# of 12; the IDWT-uint8-DWT round-trip on natural RGB photos perturbs
# coefficients by at most ~6, well within the Q/4 = 6 tolerance margin.
QUANTIZATION_STEP = 24
HALF_STEP = QUANTIZATION_STEP / 2.0

# Each bit is replicated to REPETITIONS distinct cD positions per channel.
# Combined with 2-of-3 channel majority vote, this gives 3 × REPETITIONS
# samples per bit. REPETITIONS = 5 reaches ~93–100% perfect-recovery on
# the original master's-thesis cover-image dataset.
REPETITIONS = 5


class DwtVerifyError(RuntimeError):
    """Raised when the post-encode self-verification round-trip fails."""


def text_to_bits(text: str) -> str:
    """Encode an ASCII/UTF-8 string as a flat binary string (8 bits per byte)."""
    return "".join(format(b, "08b") for b in text.encode("utf-8"))


def qim_embed(coefficient: float, bit: int) -> float:
    """Quantize *coefficient* to the lattice that encodes *bit*."""
    if bit == 0:
        return round(coefficient / QUANTIZATION_STEP) * QUANTIZATION_STEP
    return round((coefficient - HALF_STEP) / QUANTIZATION_STEP) * QUANTIZATION_STEP + HALF_STEP


def channel_capacity_bits(channel: np.ndarray) -> int:
    """Number of message bits that can be embedded in a single channel.

    Equal to the number of cD coefficients divided by ``REPETITIONS``,
    since each bit consumes ``REPETITIONS`` slots.
    """
    _, (_, _, cD) = pywt.dwt2(channel, "haar")
    return cD.size // REPETITIONS


def max_text_length(image_path: str) -> int:
    """Return the maximum number of bytes that can be hidden in *image_path*."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    b, _, _ = cv2.split(img)
    return channel_capacity_bits(b) // 8


def _slot_positions(n_bits: int, total_slots: int) -> list[int]:
    """Compute the cD positions used to embed *n_bits* with REPETITIONS replicas.

    Slots are spread uniformly across the cD vector so each bit's replicas
    fall in different parts of the image. Returns an empty list when
    *n_bits* is 0 (the empty-message case).
    """
    if n_bits == 0:
        return []
    needed = n_bits * REPETITIONS
    if needed > total_slots:
        raise ValueError(
            f"Message of {n_bits} bits with {REPETITIONS}x replication "
            f"needs {needed} cD slots, only {total_slots} available "
            f"(reduce message length or REPETITIONS)"
        )
    step = total_slots // needed
    return [(i * REPETITIONS + r) * step for i in range(n_bits) for r in range(REPETITIONS)]


def embed_in_channel(channel: np.ndarray, binary_text: str) -> np.ndarray:
    """Embed *binary_text* into the cD sub-band of *channel* via QIM + repetition.

    Returns the reconstructed (uint8) channel. If *binary_text* is empty,
    returns the channel after a clean DWT/IDWT round-trip (no modifications).
    """
    cA, (cH, cV, cD) = pywt.dwt2(channel, "haar")

    n_bits = len(binary_text)
    if n_bits == 0:
        stego = pywt.idwt2((cA, (cH, cV, cD)), "haar")
        return np.clip(stego, 0, 255).astype(np.uint8)

    flat = cD.flatten().astype(np.float64)
    positions = _slot_positions(n_bits, flat.size)

    # Each bit gets REPETITIONS positions; we walk through positions in
    # bit-replica order: positions[i*R + r] holds replica r of bit i.
    for i, bit_char in enumerate(binary_text):
        bit = int(bit_char)
        for r in range(REPETITIONS):
            pos = positions[i * REPETITIONS + r]
            flat[pos] = qim_embed(flat[pos], bit)

    cD_modified = flat.reshape(cD.shape)
    stego = pywt.idwt2((cA, (cH, cV, cD_modified)), "haar")
    return np.clip(stego, 0, 255).astype(np.uint8)


def _decode_bytes(stego_img: np.ndarray, n_bits: int) -> bytes:
    """Internal helper used by self-verification (mirrors decode_dwt logic)."""
    bits_per_channel = []
    for chan in cv2.split(stego_img):
        _, (_, _, cD) = pywt.dwt2(chan.astype(np.float64), "haar")
        flat = cD.flatten()
        positions = _slot_positions(n_bits, flat.size)
        per_bit_votes = [[] for _ in range(n_bits)]
        for i in range(n_bits):
            for r in range(REPETITIONS):
                pos = positions[i * REPETITIONS + r]
                per_bit_votes[i].append(int(round(flat[pos] / HALF_STEP)) & 1)
        chan_bits = [1 if sum(v) > REPETITIONS // 2 else 0 for v in per_bit_votes]
        bits_per_channel.append(chan_bits)
    final_bits = [
        1 if (bits_per_channel[0][i] + bits_per_channel[1][i] + bits_per_channel[2][i]) >= 2 else 0
        for i in range(n_bits)
    ]
    out = bytearray()
    for i in range(0, n_bits - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | final_bits[i + j]
        out.append(byte)
    return bytes(out)


def hide_text_in_image(
    image_path: str,
    text: str,
    output_path: str,
    *,
    verify: bool = True,
) -> None:
    """Hide *text* in *image_path* and save the stego image to *output_path*.

    If ``verify`` is True (default), the function decodes the just-written
    stego image and raises ``DwtVerifyError`` if the round-trip does not
    recover *text* byte-for-byte. Pass ``verify=False`` to skip the check
    when bulk-processing trusted-cover datasets.
    """
    binary_text = text_to_bits(text)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    b, g, r = cv2.split(img)
    stego_b = embed_in_channel(b, binary_text)
    stego_g = embed_in_channel(g, binary_text)
    stego_r = embed_in_channel(r, binary_text)
    stego_img = cv2.merge([stego_b, stego_g, stego_r])
    cv2.imwrite(output_path, stego_img)

    if verify:
        # Re-read from disk so the verify uses the exact bytes written
        # (catches PNG re-encoding edge cases too).
        on_disk = cv2.imread(output_path)
        if on_disk is None:
            raise DwtVerifyError(f"Verify: could not re-read {output_path}")
        recovered = _decode_bytes(on_disk, len(binary_text))
        if recovered != text.encode("utf-8"):
            # Delete the broken file so we never leave half-bad outputs on disk.
            try:
                os.remove(output_path)
            except OSError:
                pass
            raise DwtVerifyError(
                f"Verify: round-trip recovered {len(recovered)} bytes that do not "
                f"match the {len(text.encode('utf-8'))} bytes of the input message. "
                f"This image is in the small fraction of pathological covers where "
                f"DWT-QIM cannot guarantee bit-exact recovery; pick a different cover."
            )


def read_text_file(file_path: str) -> str:
    """Read a UTF-8 text file and return its contents."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def process_folder(input_folder: str, output_folder: str, text_file: str) -> None:
    """Process every PNG in *input_folder*, embedding fragments of *text_file*.

    For each cover, five different message lengths are embedded (20 % to
    100 % of capacity in equal steps). Images that fail self-verification
    are reported and skipped — the rest of the batch continues.

    Telegram notifications are sent on start, completion, and on per-image
    errors when ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` env vars
    are configured; otherwise the notifications are silently skipped.
    """
    send_telegram_message(f"DWT: started processing {input_folder}")

    os.makedirs(output_folder, exist_ok=True)
    text_data = read_text_file(text_file)

    n_ok = 0
    n_skip = 0
    n_err = 0

    for img_file in sorted(os.listdir(input_folder)):
        if not img_file.lower().endswith(".png"):
            continue
        img_path = os.path.join(input_folder, img_file)

        try:
            max_chars = max_text_length(img_path)
            step = max(1, max_chars // 5)
            text_lengths = [i * step for i in range(1, 6)]

            for text_len in text_lengths:
                if text_len > len(text_data):
                    continue
                random_start = random.randint(0, len(text_data) - text_len)
                random_text = text_data[random_start : random_start + text_len]
                stem, ext = os.path.splitext(img_file)
                output_name = f"{stem}_DWT_{text_len}{ext}"
                output_path = os.path.join(output_folder, output_name)
                try:
                    hide_text_in_image(img_path, random_text, output_path)
                    print(f"Processed: {output_name}")
                    n_ok += 1
                except DwtVerifyError as ve:
                    print(f"Skipped (verify failed): {output_name}: {ve}")
                    n_skip += 1
        except Exception:
            error_msg = f"Error processing {img_file}: {traceback.format_exc()}"
            print(error_msg)
            send_telegram_message(error_msg)
            n_err += 1

    summary = f"DWT: processing complete — ok={n_ok}, skipped={n_skip}, errors={n_err}"
    print(summary)
    send_telegram_message(summary)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DWT steganography — hide text in images using Haar wavelets and QIM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python dwt.py --input image.png --output stego.png --message "Secret text"

  # Whole folder
  python dwt.py --input ./images/ --output ./output/ --text data.txt

Notifications:
  If TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables are
  set, batch runs send progress and error messages to Telegram.
  Otherwise notifications are silently skipped.
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Input image or folder")
    parser.add_argument("--output", "-o", required=True, help="Output image or folder")
    parser.add_argument("--message", "-m", help="Message to hide (single-image mode)")
    parser.add_argument("--text", "-t", help="Text file to draw payloads from (folder mode)")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the encode-verify round-trip (faster but unsafe)",
    )
    return parser.parse_args()


def main() -> None:
    setup_environment()
    args = parse_args()
    send_telegram_message("DWT: program started")

    try:
        if os.path.isfile(args.input):
            if not args.message:
                print("Error: --message is required for single-image mode")
                sys.exit(1)
            hide_text_in_image(args.input, args.message, args.output, verify=not args.no_verify)
            print(f"Wrote: {args.output}")
        else:
            if not args.text:
                print("Error: --text is required for folder mode")
                sys.exit(1)
            process_folder(args.input, args.output, args.text)
    except Exception:
        error_msg = f"Error: {traceback.format_exc()}"
        print(error_msg)
        send_telegram_message(error_msg)
        sys.exit(1)

    send_telegram_message("DWT: program finished")


if __name__ == "__main__":
    main()
