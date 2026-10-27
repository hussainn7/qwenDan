"""HTTP client for the daemon; starts it in the background when it isn't running."""

import os
import subprocess
import sys
import time

import httpx

from .voices import HOME

PORT = int(os.environ.get("QWEN_VOICE_PORT", "7731"))
BASE_URL = f"http://127.0.0.1:{PORT}"
LOG = HOME / "daemon.log"


def health():
    try:
        return httpx.get(f"{BASE_URL}/health", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return None


def ensure_daemon(timeout_s=120):
    if health():
        return
    HOME.mkdir(parents=True, exist_ok=True)
    with open(LOG, "ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "qwen_voice.server"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=HOME,
        )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if health():
            return
    raise RuntimeError(f"qwen-voice daemon did not start; see {LOG}")


def _call(method, path, **kw):
    ensure_daemon()
    r = httpx.request(method, f"{BASE_URL}{path}", timeout=None, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"qwen-voice: {detail}")
    return r.json()


def speak(text, **options):
    payload = {"text": text, **{k: v for k, v in options.items() if v is not None}}
    return _call("POST", "/speak", json=payload)


def speak_batch(texts, **options):
    """Several short texts in one generation pass; raises RuntimeError on daemons without /speak_batch."""
    payload = {"texts": list(texts), **{k: v for k, v in options.items() if v is not None}}
    return _call("POST", "/speak_batch", json=payload)


def stop():
    if not health():
        return False
    httpx.post(f"{BASE_URL}/shutdown", timeout=5)
    return True
