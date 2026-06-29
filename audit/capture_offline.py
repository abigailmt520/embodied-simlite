# -*- coding: utf-8 -*-
"""
capture_offline.py  ——  Action1 · 1-C real-browser OFFLINE screenshot (headless Chrome via CDP)
======================================================================================
Drive the **real** inference_server frontend, reproduce "WS disconnect → frontend freezes and shows OFFLINE" and screenshot it:
    1) launch the real inference_server (without changing a line of frontend code);
    2) headless Chrome opens the page, waits for WS to connect, the view to render, status=online;
    3) **kill the server** to create a real disconnect → triggers the frontend ws.onclose → setOffline() freeze overlay;
    4) save the screenshot.

This captures the real frontend's real disconnect behavior, not a faked page (upholding INV-2).
Dependencies: only system Chrome + the standard library + websockets (CDP communication), no selenium/playwright.

Run: python audit/capture_offline.py
Artifact: audit/offline_screenshot.png
"""

import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SERVER_PORT = 8000
CDP_PORT = 9222
URL = f"http://localhost:{SERVER_PORT}/"
OUT = os.path.join(HERE, "offline_screenshot.png")


def _wait_http(url, timeout=25):
    for _ in range(int(timeout * 2)):
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self._id = 0

    async def cmd(self, method, **params):
        self._id += 1
        mid = self._id
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method} error: {msg['error']}")
                return msg.get("result", {})

    async def eval_js(self, expr):
        r = await self.cmd("Runtime.evaluate", expression=expr, returnByValue=True)
        return r.get("result", {}).get("value")


async def drive(server_proc):
    # get the page target's CDP ws address
    targets = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:{CDP_PORT}/json").read().decode())
    page = next((t for t in targets if t.get("type") == "page"), targets[0])
    ws_url = page["webSocketDebuggerUrl"]

    async with websockets.connect(ws_url, max_size=None) as ws:
        cdp = CDP(ws)
        await cdp.cmd("Page.enable")
        await cdp.cmd("Runtime.enable")
        await cdp.cmd("Emulation.setDeviceMetricsOverride",
                      width=1600, height=900, deviceScaleFactor=1, mobile=False)
        await cdp.cmd("Page.navigate", url=URL)

        # wait for WS to connect, the view online (wsStatus contains "online")
        online = False
        for _ in range(30):
            await asyncio.sleep(0.5)
            txt = await cdp.eval_js(
                "(document.getElementById('wsStatus')||{}).innerText || ''")
            if txt and ("online" in txt.lower()):
                online = True
                break
        print(f"  [chrome] connection status text: online={online}")
        await asyncio.sleep(1.5)   # receive a few more frames so the robot/lidar rendering stabilizes

        # —— create a real disconnect: kill the server ——
        print("  [server] killing the server to create a real WS disconnect…")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()

        # wait for the frontend onclose → the OFFLINE overlay appears
        offline = False
        for _ in range(20):
            await asyncio.sleep(0.4)
            disp = await cdp.eval_js(
                "(document.getElementById('offline-overlay')||{}).style?"
                ".display || ''")
            status = await cdp.eval_js(
                "(document.getElementById('wsStatus')||{}).innerText || ''")
            epi = await cdp.eval_js(
                "(document.getElementById('epi_status')||{}).innerText || ''")
            if disp == "flex" or "OFFLINE" in (status or "") or "OFFLINE" in (epi or ""):
                offline = True
                print(f"  [chrome] OFFLINE triggered: overlay.display={disp!r} "
                      f"status={status!r} epi={epi!r}")
                break
        if not offline:
            print("  [warn] OFFLINE state not detected (still capturing the current frame for inspection)")
        await asyncio.sleep(0.8)

        shot = await cdp.cmd("Page.captureScreenshot", format="png",
                             captureBeyondViewport=False)
        with open(OUT, "wb") as f:
            f.write(base64.b64decode(shot["data"]))
        print(f"  [OK] screenshot saved: {OUT}")
        return offline


def main():
    if not os.path.exists(CHROME):
        print(f"[ERR] Chrome not found: {CHROME}")
        sys.exit(1)

    # 1) launch the real server
    print("[1/3] launching the real inference_server …")
    server = subprocess.Popen([sys.executable, "inference_server.py"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_http(f"http://127.0.0.1:{SERVER_PORT}/health"):
        server.kill()
        print("[ERR] server not ready")
        sys.exit(1)
    print("      server ready.")

    # 2) launch headless Chrome (separate user-data-dir, with remote debugging)
    print("[2/3] launching headless Chrome …")
    profile = tempfile.mkdtemp(prefix="cdp_chrome_")
    chrome = subprocess.Popen([
        CHROME, "--headless=new", f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={profile}", "--window-size=1600,900",
        "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
        "--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader",
        "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_http(f"http://127.0.0.1:{CDP_PORT}/json/version"):
        chrome.kill(); server.kill(); shutil.rmtree(profile, ignore_errors=True)
        print("[ERR] Chrome CDP not ready")
        sys.exit(1)

    # 3) drive + screenshot
    print("[3/3] driving the page, disconnecting, capturing …")
    ok = False
    try:
        ok = asyncio.run(drive(server))
    finally:
        for p in (server, chrome):
            try:
                p.terminate(); p.wait(timeout=3)
            except Exception:
                p.kill()
        shutil.rmtree(profile, ignore_errors=True)
    print(f"\nresult: OFFLINE screenshot {'succeeded' if ok else 'generated (please manually verify OFFLINE is visible)'} → {OUT}")


if __name__ == "__main__":
    main()
