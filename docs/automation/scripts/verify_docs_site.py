#!/usr/bin/env python3
"""Browser-verify the deployed documentation layout end to end.

Builds a synthetic GitHub-Pages-shaped tree by running the real deploy script,
serves it the way GitHub Pages does, then drives a browser through:

  * version-switcher population from versions.json
  * path-preserving version switching (deep page -> deep page)
  * fallback to version root when the deep page is absent in the target
  * 404 handler rewriting legacy flat deep links
  * 404 handler leaving URLs that are already versioned alone

Run: python3 docs/automation/scripts/verify_docs_site.py
Exit 0 = all checks passed.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DEPLOY = HERE / "deploy_unified_docs.sh"
DOCS = REPO / "docs" / "source"

PAGE = """<!DOCTYPE html><html><head><title>{title}</title>
<script src="{rel}_static/version_switch.js"></script></head><body>
<h1>{title}</h1>
<select id="sel" data-docs-version></select>
</body></html>"""

# A page present in BOTH versions (deep-link rewriting must preserve it).
DEEP = "python_api/index.html"
# A page present in the archive ONLY (forces the version-root fallback).
ARCHIVE_ONLY = "guide/advanced.html"


def write_page(base: Path, relative: str, title: str) -> None:
    """Write a Sphinx-shaped page: correct per-page relative asset depth."""
    target = base / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    depth = relative.count("/")
    rel = "../" * depth
    target.write_text(PAGE.format(title=title, rel=rel))


def build_tree(root: Path) -> None:
    """Run the real deploy script against a synthetic build."""
    (root / "website-deploy").mkdir(parents=True)
    shutil.copy(DOCS / ".." / "404.html", root / "website-deploy" / "404.html")
    (root / "website-deploy" / "index.html").write_text("<html>site</html>")

    dev = root / "docs-output"
    write_page(dev, "index.html", "dev home")
    write_page(dev, DEEP, "dev python api")
    (dev / "_static").mkdir(parents=True, exist_ok=True)
    shutil.copy(DOCS / "_static" / "version_switch.js",
                dev / "_static" / "version_switch.js")

    # One archived release already restored from GitHub Releases.
    arch = root / "preserved-versions" / "documentation" / "v3.9.4"
    write_page(arch, "index.html", "v3.9.4 home")
    write_page(arch, DEEP, "v3.9.4 python api")
    write_page(arch, ARCHIVE_ONLY, "v3.9.4 advanced guide")
    (arch / "_static").mkdir(parents=True, exist_ok=True)
    (arch / "_static" / "version_switch.js").write_text("/* stale switcher */")

    subprocess.run(
        ["bash", str(DEPLOY)],
        cwd=root,
        env={**os.environ, "IS_RELEASE": "false", "RELEASE_VERSION": "", "DOC_VERSION": ""},
        check=True,
        capture_output=True,
    )


class PagesLikeHandler(http.server.SimpleHTTPRequestHandler):
    """Serve like GitHub Pages: unknown paths yield the 404 page AND a 404.

    The 404 status matters: the switcher probes a candidate URL with HEAD to
    decide whether the reader's current page exists in the target version.
    A 200-for-everything server would make that probe meaningless.
    """

    def log_message(self, *args):
        pass

    def send_head(self):
        requested = self.translate_path(self.path)
        if not os.path.exists(requested):
            self.send_response(404)
            body = (Path(self.directory) / "404.html").read_bytes()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(body)
            return None
        return super().send_head()


def serve(directory: Path):
    handler = functools.partial(PagesLikeHandler, directory=str(directory))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


class Recorder:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed.append(name)
            print(f"  ❌ {name}" + (f"  [{detail}]" if detail else ""))


def options(page):
    return page.eval_on_selector_all(
        "#sel option", "els => els.map(e => [e.value, e.textContent.trim(), e.selected])")


def main() -> int:
    root = Path(tempfile.mkdtemp())
    try:
        build_tree(root)
        deploy = root / "unified-deploy"
        doc = deploy / "documentation"
        rec = Recorder()

        print("═══ artifact contract ═══")
        manifest = json.loads((doc / "versions.json").read_text())
        ids = [v["id"] for v in manifest["versions"]]
        rec.check("versions.json lists dev + v3.9.4", ids == ["dev", "v3.9.4"], str(ids))
        rec.check("latest resolves to v3.9.4", manifest["latest"] == "v3.9.4", manifest["latest"])
        rec.check("root redirect targets v3.9.4",
                  "./v3.9.4/index.html" in (doc / "index.html").read_text())
        rec.check("archive received the current switcher",
                  "versions.json" in (doc / "v3.9.4" / "_static" / "version_switch.js").read_text())
        rec.check("dev archive untouched by propagation",
                  "versions.json" in (doc / "dev" / "_static" / "version_switch.js").read_text())
        rec.check("404 calibrated to v3.9.4",
                  "var version = 'v3.9.4'" in (deploy / "404.html").read_text())

        httpd, base = serve(deploy)
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                ctx = browser.new_context()

                def goto(path: str):
                    page = ctx.new_page()
                    page.goto(base + path, wait_until="domcontentloaded")
                    return page

                print("═══ switcher population ═══")
                page = goto("/documentation/dev/")
                page.wait_for_selector("#sel:not([disabled])", timeout=5000)
                opts = options(page)
                rec.check("dev page offers both versions",
                          [o[0] for o in opts] == ["dev", "v3.9.4"], str(opts))
                rec.check("current version selected",
                          any(o[0] == "dev" and o[2] for o in opts), str(opts))
                rec.check("newest release labelled as latest stable",
                          any(o[0] == "v3.9.4" and "latest stable" in o[1] for o in opts),
                          str(opts))

                page = goto("/documentation/v3.9.4/")
                page.wait_for_selector("#sel:not([disabled])", timeout=5000)
                opts = options(page)
                rec.check("archive page marks its own version selected",
                          any(o[0] == "v3.9.4" and o[2] for o in opts), str(opts))

                print("═══ path-preserving switch ═══")
                page.goto(base + "/documentation/dev/" + DEEP, wait_until="domcontentloaded")
                page.wait_for_selector("#sel:not([disabled])", timeout=5000)
                page.select_option("#sel", "v3.9.4")
                page.wait_for_url("**/documentation/v3.9.4/" + DEEP, timeout=8000)
                rec.check("deep page maps to the same deep page",
                          page.url.endswith("/documentation/v3.9.4/" + DEEP), page.url)
                rec.check("landed on the archive content", "v3.9.4 python api" in page.content())

                print("═══ missing-page fallback ═══")
                page.goto(base + "/documentation/v3.9.4/" + ARCHIVE_ONLY,
                          wait_until="domcontentloaded")
                page.wait_for_selector("#sel:not([disabled])", timeout=5000)
                page.select_option("#sel", "dev")
                page.wait_for_url("**/documentation/dev/", timeout=8000)
                rec.check("absent deep page falls back to the version root",
                          page.url.rstrip("/").endswith("/documentation/dev"), page.url)

                print("═══ 404 handler ═══")
                page = goto("/documentation/getting_started/build_from_source.html")
                page.wait_for_timeout(800)
                rec.check("legacy flat deep link rewritten to the newest version",
                          "/documentation/v3.9.4/getting_started/build_from_source.html"
                          in page.url, page.url)

                page = goto("/documentation/v9.9.9/nope.html")
                page.wait_for_timeout(400)
                rec.check("already-versioned URL is left alone",
                          "/documentation/v9.9.9/nope.html" in page.url, page.url)

                page = goto("/documentation/dev/nope.html")
                page.wait_for_timeout(400)
                rec.check("versioned dev miss is left alone",
                          "/documentation/dev/nope.html" in page.url, page.url)

                browser.close()
        finally:
            httpd.shutdown()

        print()
        print("──────────────────────────────")
        print(f"PASS: {rec.passed}   FAIL: {len(rec.failed)}")
        for name in rec.failed:
            print(f"  failed: {name}")
        return 0 if not rec.failed else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
