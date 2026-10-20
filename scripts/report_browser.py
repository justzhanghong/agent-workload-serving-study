#!/usr/bin/env python3
"""Owned, isolated headless Chrome for local report assets; no user profile used."""
import base64
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from urllib.request import urlopen
import websocket

ROOT = Path(__file__).resolve().parents[1]


class ReportBrowser:
    def __enter__(self):
        binary = os.environ.get("AGENT_STUDY_CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
        if not binary:
            raise RuntimeError("Chrome/Chromium is required; set AGENT_STUDY_CHROME")
        self.version = subprocess.check_output([binary, "--version"], text=True).strip()
        delivery = ROOT / ".delivery"
        delivery.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="report-chrome-", dir=delivery)
        self.process = subprocess.Popen([
            binary, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
            "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0",
            "--remote-allow-origins=http://localhost", "--user-data-dir="+self.temp.name,
            "--disable-background-networking", "about:blank",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.ws, self.counter = None, 0
        try:
            deadline = time.monotonic()+25
            active = Path(self.temp.name) / "DevToolsActivePort"
            while not active.exists():
                if self.process.poll() is not None:
                    raise RuntimeError("Owned report Chrome exited during startup")
                if time.monotonic() > deadline:
                    raise TimeoutError("Owned report Chrome did not become ready")
                time.sleep(.1)
            port = int(active.read_text().splitlines()[0])
            with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as response:
                page = next(p for p in json.load(response) if p["type"] == "page")
            self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], origin="http://localhost", timeout=45)
            self.call("Page.enable")
            self.call("Runtime.enable")
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise

    def call(self, method, params=None):
        self.counter += 1
        ident = self.counter
        self.ws.send(json.dumps({"id":ident, "method":method, "params":params or {}}))
        while True:
            value = json.loads(self.ws.recv())
            if value.get("id") == ident:
                if "error" in value:
                    raise RuntimeError(str(value["error"]))
                return value.get("result", {})

    def navigate(self, path, width=1500, height=1000):
        self.call("Emulation.setDeviceMetricsOverride", {"width":width,"height":height,"deviceScaleFactor":1,"mobile":False})
        self.call("Page.navigate", {"url":Path(path).resolve().as_uri()})
        deadline = time.monotonic()+30
        while True:
            result = self.call("Runtime.evaluate", {"expression":"document.readyState === 'complete' && document.fonts.status === 'loaded' && [...document.images].every(i => i.complete && i.naturalWidth > 0)", "returnByValue":True})
            if result.get("result",{}).get("value"):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Report fonts/images did not finish loading")
            time.sleep(.1)

    def screenshot(self, source, target, width, height):
        self.navigate(source, width, height)
        result = self.call("Page.captureScreenshot", {"format":"png","clip":{"x":0,"y":0,"width":width,"height":height,"scale":1.5}})
        Path(target).write_bytes(base64.b64decode(result["data"]))

    def pdf(self, source):
        self.navigate(source)
        return base64.b64decode(self.call("Page.printToPDF", {
            "printBackground":True, "preferCSSPageSize":True,
            "generateTaggedPDF":True, "generateDocumentOutline":True,
            "displayHeaderFooter":True, "headerTemplate":"<span></span>",
            "footerTemplate":"<div style='font-size:8px;width:100%;text-align:center;color:#777'><span class='pageNumber'></span> / <span class='totalPages'></span></div>",
        })["data"])

    def __exit__(self, *_):
        if self.ws is not None:
            self.ws.close()
        if self.process.poll() is None:
            # This process group was created by this context, never a user browser.
            os.killpg(self.process.pid, signal.SIGTERM)
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self.temp.cleanup()
