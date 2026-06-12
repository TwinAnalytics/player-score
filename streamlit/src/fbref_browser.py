# src/fbref_browser.py
"""
Shared browser session for the FBref scrapers.

FBref sits behind a Cloudflare bot challenge (since spring 2026) that blocks
both headless and regular automated Chrome (CDP detection). Patchright — a
stealth-patched Playwright fork — combined with the real Chrome channel, a
persistent profile and headed mode passes the challenge reliably.

Falls back to plain Playwright if patchright is not installed (will likely
hit the challenge again, but keeps the import safe).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

try:
    from patchright.sync_api import sync_playwright
    STEALTH = True
except ImportError:  # pragma: no cover
    from playwright.sync_api import sync_playwright
    STEALTH = False

# Persistent profile keeps the cf_clearance cookie between runs,
# so usually only the first request of a run sees the challenge.
PROFILE_DIR = Path.home() / ".cache" / "playerscore-fbref-profile"


@contextmanager
def fbref_page():
    """Yield a Playwright page suited for FBref (headed, real Chrome if available)."""
    if not STEALTH:
        print("[FBREF WARN] patchright not installed - Cloudflare challenge will likely block scraping. pip install patchright")
    with sync_playwright() as p:
        kwargs = dict(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            no_viewport=True,
        )
        try:
            ctx = p.chromium.launch_persistent_context(channel="chrome", **kwargs)
        except Exception:
            # No system Chrome installed - fall back to bundled Chromium
            ctx = p.chromium.launch_persistent_context(**kwargs)
        page = ctx.new_page()
        try:
            yield page
        finally:
            ctx.close()


def wait_for_cloudflare(page, max_wait_s: int = 45) -> None:
    """Block until the Cloudflare interstitial ('Just a moment...') is gone."""
    waited = 0
    while waited < max_wait_s:
        title = (page.title() or "").lower()
        if "moment" not in title:
            return
        page.wait_for_timeout(3000)
        waited += 3
    print(f"[FBREF WARN] Cloudflare challenge still present after {max_wait_s}s")
