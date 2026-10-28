"""Optional integration check using a disposable headless Edge profile, never your webcam.

Run a server first, then: uv run --with playwright python scripts/browser_smoke.py
"""

import json
import os
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    output = Path("outputs/browser")
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1050})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(os.environ.get("DROWSINESS_TEST_URL", "http://127.0.0.1:8000"))
        expect(page.locator("#runtime")).to_have_text(re.compile("local only"))
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1050})
        page.get_by_role("button", name="Try a simulated session").click()
        expect(page.locator("body")).to_have_attribute("data-state", "critical", timeout=30000)
        assert page.locator("#demo-banner").is_visible()
        assert page.locator("#score").inner_text() != "—"
        page.screenshot(path=str(output / "critical-demo.png"), full_page=True)
        with page.expect_download() as download:
            page.get_by_role("button", name="Export session JSON").click()
        download.value.save_as(str(output / "session.json"))
        session = json.loads((output / "session.json").read_text())
        assert session["mode"] == "demo" and len(session["samples"]) > 10
        assert session["metadata"]["baseline"] is not None
        page.get_by_role("button", name="Stop session").click()
        assert page.get_by_role("button", name="Start camera").is_enabled()
        page.get_by_role("button", name="Start camera").click()
        expect(page.locator("#reason")).to_have_text(
            re.compile("Calibration paused|No face detected"), timeout=30000
        )
        assert page.locator("#video").is_visible()
        page.get_by_role("button", name="Stop session").click()
        assert page.evaluate("document.getElementById('video').srcObject === null")
        assert not errors, errors
        browser.close()
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "page_errors": errors,
                "checks": [
                    "desktop",
                    "mobile",
                    "no overflow",
                    "demo critical warning",
                    "JSON export",
                    "fake camera start",
                    "no-face calibration pause",
                    "camera stop",
                ],
            },
            indent=2,
        )
    )
    print("Browser smoke passed; screenshots and session export saved under outputs/browser")


if __name__ == "__main__":
    main()
