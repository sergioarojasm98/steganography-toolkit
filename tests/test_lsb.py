"""Round-trip tests for the LSB steganography method."""

from __future__ import annotations

from pathlib import Path

from lsb import decode_lsb, lsb


def _round_trip(cover_path: Path, message: str, tmp_path: Path) -> str:
    stego_path = tmp_path / "stego.png"
    lsb.hide_text_in_image(str(cover_path), message, str(stego_path))
    return decode_lsb.decode_text_from_image(str(stego_path))


def test_round_trip_short_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    msg = "Hello"
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_medium_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    msg = "The quick brown fox jumps over the lazy dog"
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_long_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    """LSB has very high capacity (~3 bits per pixel)."""
    msg = "X" * 5000
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_on_smooth(smooth_cover: Path, tmp_path: Path) -> None:
    msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    assert _round_trip(smooth_cover, msg, tmp_path) == msg


def test_round_trip_on_small_image(small_cover: Path, tmp_path: Path) -> None:
    msg = "Hi"
    assert _round_trip(small_cover, msg, tmp_path) == msg


def test_round_trip_empty_message(noise_cover: Path, tmp_path: Path) -> None:
    """Encoding an empty payload still produces a valid stego image."""
    msg = ""
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_capacity_is_per_byte(noise_cover: Path) -> None:
    """LSB capacity should be ~H*W*3/8 (3 channels, 1 bit per channel per pixel).

    The actual reported capacity reserves one byte for the null terminator,
    so it is ``(H*W*3) // 8 - 1``.
    """
    import cv2

    img = cv2.imread(str(noise_cover))
    h, w, _ = img.shape
    expected = (h * w * 3) // 8 - 1  # -1 byte for the \0 end-of-message marker
    assert lsb.max_text_length(str(noise_cover)) == expected


def test_round_trip_at_quarter_capacity(noise_cover: Path, tmp_path: Path) -> None:
    """A message at 25% of reported capacity should round-trip cleanly."""
    cap = lsb.max_text_length(str(noise_cover))
    msg = "X" * (cap // 4)
    assert _round_trip(noise_cover, msg, tmp_path) == msg
