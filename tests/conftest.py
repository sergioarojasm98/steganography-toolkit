"""Shared pytest fixtures for the toolkit test suite.

Generates three deterministic synthetic cover images that exercise
different image statistics:

* ``noise_cover``     — uniform random RGB noise (high-frequency, hard for DWT)
* ``smooth_cover``    — sinusoidal gradient (low-frequency, easy)
* ``small_cover``     — 64×64 random RGB (smallest reasonable test size)

Each fixture writes its image to a per-test temporary directory and
returns the file path. Using PNG (lossless) is mandatory because the
toolkit assumes pixel-exact storage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make the toolkit importable when tests are run from any cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def noise_cover(tmp_path: Path) -> Path:
    """480x360 RGB image filled with deterministic random noise."""
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, (360, 480, 3), dtype=np.uint8)
    path = tmp_path / "noise.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def smooth_cover(tmp_path: Path) -> Path:
    """480x360 RGB image with a smooth sinusoidal gradient."""
    xx, yy = np.meshgrid(np.arange(480), np.arange(360))
    grad = (np.sin(xx / 30) * np.cos(yy / 40) * 60 + 128).astype(np.uint8)
    img = np.stack([grad, np.clip(grad + 10, 0, 255), np.clip(grad - 10, 0, 255)], axis=-1)
    path = tmp_path / "smooth.png"
    cv2.imwrite(str(path), img.astype(np.uint8))
    return path


@pytest.fixture
def small_cover(tmp_path: Path) -> Path:
    """64x64 RGB image with deterministic random noise."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    path = tmp_path / "small.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture(autouse=True)
def _silence_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make absolutely sure no test ever tries to hit the Telegram API."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
