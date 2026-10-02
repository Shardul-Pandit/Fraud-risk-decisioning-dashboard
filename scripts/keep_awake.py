"""
Keep the Streamlit Community Cloud app awake.

Why a browser and not a plain HTTP request: a sleeping Streamlit app still
answers HTTP 200 with a "this app has gone to sleep" page. The app only
starts again when someone clicks the wake-up button, and it only counts as
"visited" when a real browser opens the websocket. So this script opens the
app in headless Chromium, clicks the button if it is there, and then waits
until the real app has rendered.

Exits with code 1 if the app never comes up, so the GitHub Action fails
and GitHub emails you.
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("APP_URL", "https://fraudguard-dashboard.streamlit.app")
APP_MARKER = os.environ.get("APP_MARKER", "Fraud Risk Decisioning Dashboard")
WAKE_BUTTON = re.compile("get this app back up", re.IGNORECASE)
TIMEOUT_SECONDS = int(os.environ.get("WAKE_TIMEOUT_SECONDS", "300"))


def app_is_rendered(page) -> bool:
    # Streamlit Cloud serves the app inside an iframe, so check every frame.
    for frame in page.frames:
        try:
            if frame.get_by_text(APP_MARKER).count() > 0:
                return True
        except Exception:
            continue
    return False


def main() -> int:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)

        deadline = time.time() + TIMEOUT_SECONDS
        clicked = False

        while time.time() < deadline:
            if app_is_rendered(page):
                print("App was asleep and is awake now." if clicked else "App is awake.")
                # Stay connected briefly so the visit registers.
                page.wait_for_timeout(5_000)
                browser.close()
                return 0

            button = page.get_by_role("button", name=WAKE_BUTTON)
            if not clicked and button.count() > 0:
                print("App is asleep. Clicking the wake-up button.")
                button.first.click()
                clicked = True

            page.wait_for_timeout(3_000)

        print(f"App did not render within {TIMEOUT_SECONDS} seconds.")
        page.screenshot(path="keep_awake_failure.png")
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
