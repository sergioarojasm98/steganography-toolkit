"""Round-trip tests for the DCT-QIM steganography method.

The DCT module uses QIM + 5x repetition on the first 20 AC coefficients
of each 8x8 DCT block, independently in all three BGR channels with
2-of-3 channel majority voting at decode time.
"""

from __future__ import annotations

from pathlib import Path

from dct import dct


def _round_trip(cover_path: Path, message: str, tmp_path: Path) -> str:
    stego_path = tmp_path / "stego.png"
    dct.hide_text_in_image(str(cover_path), message, str(stego_path))
    return dct.decode_from_file(str(stego_path), text_length=len(message))


def test_round_trip_short_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    msg = "Hello"
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_medium_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    msg = "The quick brown fox jumps over the lazy dog"
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_long_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    msg = "A" * 500
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_on_smooth(smooth_cover: Path, tmp_path: Path) -> None:
    msg = "Hello World"
    assert _round_trip(smooth_cover, msg, tmp_path) == msg


def test_round_trip_special_characters(noise_cover: Path, tmp_path: Path) -> None:
    msg = "!@#$%^&*() 0123456789"
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_empty_message(noise_cover: Path, tmp_path: Path) -> None:
    stego = tmp_path / "stego.png"
    dct.hide_text_in_image(str(noise_cover), "", str(stego))
    assert stego.exists()
    assert dct.decode_from_file(str(stego), text_length=0) == ""


def test_capacity_reasonable_for_480x360(noise_cover: Path) -> None:
    """With QIM + 5x rep on 20 AC positions, capacity should be ~1350 chars."""
    cap = dct.max_text_length(str(noise_cover))
    assert cap > 1000, f"expected >1000 chars, got {cap}"


def test_image_dimensions_preserved(noise_cover: Path, tmp_path: Path) -> None:
    import cv2

    stego = tmp_path / "stego.png"
    dct.hide_text_in_image(str(noise_cover), "Hello", str(stego))
    assert cv2.imread(str(noise_cover)).shape == cv2.imread(str(stego)).shape


def test_at_half_capacity(noise_cover: Path, tmp_path: Path) -> None:
    cap = dct.max_text_length(str(noise_cover))
    msg = "X" * (cap // 2)
    assert _round_trip(noise_cover, msg, tmp_path) == msg
