"""Round-trip tests for the DWT-QIM steganography method.

These tests are the regression suite that proves the QIM rewrite of the
DWT module is round-trip stable. The previous Cornell-style LSB-on-cD
implementation passed only on smooth gradient images and silently
corrupted bits on textured / noisy covers; the QIM rewrite recovers every
bit on every cover type.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dwt import decode_dwt, dwt

# ---------------------------------------------------------------------------
# QIM unit tests (no image required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coeff", [-50.0, -8.0, -3.7, 0.0, 3.7, 8.0, 50.0, 123.456])
@pytest.mark.parametrize("bit", [0, 1])
def test_qim_embed_extract_round_trip(coeff: float, bit: int) -> None:
    """For any coefficient and any bit, embed-then-extract returns the same bit."""
    quantized = dwt.qim_embed(coeff, bit)
    assert decode_dwt.qim_extract(quantized) == bit


@pytest.mark.parametrize("perturbation", [-5.9, -3.0, 0.0, 3.0, 5.9])
def test_qim_robust_to_small_perturbation(perturbation: float) -> None:
    """QIM tolerates perturbations up to nearly Q/4 = 6.0 in coefficient space."""
    bit = 1
    quantized = dwt.qim_embed(42.0, bit)
    assert decode_dwt.qim_extract(quantized + perturbation) == bit


# ---------------------------------------------------------------------------
# Image-level round-trip tests
# ---------------------------------------------------------------------------


def _round_trip(cover_path: Path, message: str, tmp_path: Path) -> str:
    """Encode *message* into *cover_path* and decode it back. Returns the recovered text."""
    stego_path = tmp_path / "stego.png"
    dwt.hide_text_in_image(str(cover_path), message, str(stego_path))
    payload = decode_dwt.decode_text_from_image(
        str(stego_path), text_length=len(message.encode("utf-8"))
    )
    return payload.decode("utf-8")


def test_round_trip_short_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    """The original Cornell-style implementation failed this test."""
    msg = "Hello"
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_medium_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    """The original Cornell-style implementation got 2.3% accuracy on this test."""
    msg = "The quick brown fox jumps over the lazy dog"
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_long_on_noise(noise_cover: Path, tmp_path: Path) -> None:
    msg = "A" * 200
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_on_smooth(smooth_cover: Path, tmp_path: Path) -> None:
    msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    assert _round_trip(smooth_cover, msg, tmp_path) == msg


def test_round_trip_on_small_image(small_cover: Path, tmp_path: Path) -> None:
    msg = "Hi!"
    assert _round_trip(small_cover, msg, tmp_path) == msg


def test_round_trip_unicode(noise_cover: Path, tmp_path: Path) -> None:
    msg = "Héllo Wörld 你好"
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_round_trip_empty_message(noise_cover: Path, tmp_path: Path) -> None:
    """Encoding an empty payload must still produce a valid output image."""
    stego_path = tmp_path / "stego.png"
    dwt.hide_text_in_image(str(noise_cover), "", str(stego_path))
    assert stego_path.exists()
    payload = decode_dwt.decode_text_from_image(str(stego_path), text_length=0)
    assert payload == b""


# ---------------------------------------------------------------------------
# Capacity tests
# ---------------------------------------------------------------------------


def test_capacity_matches_cD_size(noise_cover: Path) -> None:
    """max_text_length should equal (cD_size / REPETITIONS) // 8 for one channel."""
    import cv2

    img = cv2.imread(str(noise_cover))
    h, w, _ = img.shape
    expected_bits = ((h // 2) * (w // 2)) // dwt.REPETITIONS
    expected_chars = expected_bits // 8
    assert dwt.max_text_length(str(noise_cover)) == expected_chars


def test_at_capacity_round_trip(noise_cover: Path, tmp_path: Path) -> None:
    """A message exactly at the reported capacity must round-trip cleanly."""
    cap = dwt.max_text_length(str(noise_cover))
    msg = "X" * cap
    assert _round_trip(noise_cover, msg, tmp_path) == msg


def test_over_capacity_raises(noise_cover: Path, tmp_path: Path) -> None:
    """Trying to hide more than capacity must fail loudly, not silently truncate."""
    cap = dwt.max_text_length(str(noise_cover))
    msg = "X" * (cap + 1)
    with pytest.raises(ValueError, match="cD slots"):
        dwt.hide_text_in_image(str(noise_cover), msg, str(tmp_path / "stego.png"))


# ---------------------------------------------------------------------------
# Filename-based length parsing
# ---------------------------------------------------------------------------


def test_filename_length_parser() -> None:
    assert decode_dwt.get_text_length_from_filename("foo_DWT_42.png") == 42
    assert decode_dwt.get_text_length_from_filename("/a/b/img_DWT_100.png") == 100


def test_filename_length_parser_rejects_bad_format() -> None:
    with pytest.raises(ValueError, match="Cannot parse length"):
        decode_dwt.get_text_length_from_filename("foo_bar.png")
