"""Optional Telegram notifications for batch jobs.

Telegram is fully optional in this toolkit. If the environment variables
``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` are not set, every call to
``send_telegram_message`` becomes a silent no-op — no warnings, no errors,
no console noise. This keeps single-image runs and CI test runs clean.

To enable notifications:

.. code-block:: bash

    export TELEGRAM_BOT_TOKEN=your_bot_token_here
    export TELEGRAM_CHAT_ID=your_chat_id_here

Or place them in a ``.env`` file at the project root if ``python-dotenv``
is installed (see :mod:`common.config`).
"""

from __future__ import annotations

import os

import requests


def send_telegram_message(message: str) -> bool:
    """Send a message to Telegram if credentials are configured.

    Returns True if a message was sent successfully, False otherwise.
    Returns False silently (no print, no exception) when credentials are
    not configured.
    """
    api_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not api_token or not chat_id:
        return False

    api_url = f"https://api.telegram.org/bot{api_token}/sendMessage"
    try:
        response = requests.post(api_url, json={"chat_id": chat_id, "text": message}, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False


def telegram_enabled() -> bool:
    """Return True if Telegram credentials are present in the environment."""
    return bool(os.getenv("TELEGRAM_BOT_TOKEN")) and bool(os.getenv("TELEGRAM_CHAT_ID"))
