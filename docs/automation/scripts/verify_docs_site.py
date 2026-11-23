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
# Immutable v3.9.4 templates used IDs, value/display and versionsLoaded rather
# than the current data markers/id/label/docs:versionsready API. Keep these
# snippets exercising the old caller, not a rewritten modern equivalent.
LEGACY_DEEP = "legacy/index.html"
LEGACY_PAGE = """<!DOCTYPE html><html><head><title>legacy template</title>
<script src="../_static/version_switch.js"></script></head><body>
<select id="docs-version-select-sidebar"
        onchange="ACloudViewerVersionSwitcher.switchVersion(this.value)">
<option value="latest">Latest (main)</option></select>
<script>
(function() {
  function initVersionSelector() {
    const select = document.getElementById('docs-version-select-sidebar');
    if (!select) return;
    if (window.ACloudViewerVersionSwitcher) {
      window.ACloudViewerVersionSwitcher.updateSidebarSelector();
    } else {
      setTimeout(initVersionSelector, 200);
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initVersionSelector);
  } else { initVersionSelector(); }
  document.addEventListener('versionsLoaded', function() {
    window.legacyVersionsLoaded = (window.legacyVersionsLoaded || 0) + 1;
    if (window.ACloudViewerVersionSwitcher) {
      window.ACloudViewerVersionSwitcher.updateSidebarSelector();
    }
  });
})();
</script>
<select id="docs-version-select-main"
        onchange="if(window.ACloudViewerVersionSwitcher) { window.ACloudViewerVersionSwitcher.switchVersion(this.value); }">
<option value="latest">Latest (main)</option></select>
<script>
(function() {
  function updateMainSelector() {
    const select = document.getElementById('docs-version-select-main');
    if (!select || !window.ACloudViewerVersionSwitcher) {
      setTimeout(updateMainSelector, 200); return;
    }
    const versions = window.ACloudViewerVersionSwitcher.getVersions();
    const currentVersion = window.ACloudViewerVersionSwitcher.getCurrentVersion();
    if (versions.length > 0) {
      select.innerHTML = '';
      versions.forEach(v => {
        const option = document.createElement('option');
        option.value = v.value;
        option.textContent = v.display;
        if (v.value === currentVersion || (currentVersion === 'latest' && v.value === 'latest')) {
          option.selected = true;
        }
        select.appendChild(option);
      });
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateMainSelector);
  } else { updateMainSelector(); }
  document.addEventListener('versionsLoaded', updateMainSelector);
})();
</script></body></html>"""


def write_page(base: Path, relative: str, title: str) -> None:
    """Write a Sphinx-shaped page: correct per-page relative asset depth."""
    target = base / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    depth = relative.count("/")
    rel = "../" * depth
    target.write_text(PAGE.format(title=title, rel=rel))


def build_tree(root: Path) -> None:
    """Run the real deploy script against a synthetic build."""
    (root / "website-deploy" / "images").mkdir(parents=True)
    shutil.copy(DOCS / ".." / "images" / "ACloudViewer_logo_horizontal.png",
                root / "website-deploy" / "images" / "ACloudViewer_logo_horizontal.png")
    shutil.copy(DOCS / ".." / "index.html", root / "website-deploy" / "index.html")
    shutil.copy(DOCS / ".." / "styles.css", root / "website-deploy" / "styles.css")
    shutil.copy(DOCS / ".." / "script.js", root / "website-deploy" / "script.js")
    downloads = {
        "generated_at": "fixture",
        "version_metadata": [
            {"value": "main-devel", "display_name": "Beta", "is_default": True,
             "python_versions": ["3.11"], "ubuntu_versions": ["ubuntu20.04"]},
            {"value": "ubuntu-prebuild-deps", "display_name": "ubuntu-prebuild-deps",
             "is_default": False, "python_versions": [], "ubuntu_versions": []},
            {"value": "v3.9.4", "display_name": "v3.9.4", "is_default": False,
             "python_versions": ["3.11"], "ubuntu_versions": ["ubuntu20.04"]},
        ],
        "download_data": {},
    }
    for version in ("main-devel", "v3.9.4"):
        release_url = f"https://github.com/Asher-1/ACloudViewer/releases/download/{version}/ACloudViewer-3.9.4%2B0123456789abcdef-"
        package = {"url": release_url + f"{version}-windows.exe", "size": "1024.00 MB"}
        linux = {"url": release_url + f"{version}-linux.run", "size": "1024.00 MB"}
        downloads["download_data"][version] = {
            "windows": {"app": {"cpu": {"amd64": package}}},
            "linux": {"ubuntu20.04": {"app": {"cpu": {"amd64": linux}}}},
        }
    (root / "website-deploy" / "downloads_data.json").write_text(
        json.dumps(downloads), encoding="utf-8")
    shutil.copy(DOCS / ".." / "404.html", root / "website-deploy" / "404.html")

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
    for version in (dev, arch):
        legacy = version / LEGACY_DEEP
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text(LEGACY_PAGE)

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


def verify_download_availability(context, base, rec):
    """Exercise rolling assets whose OS/package branches are only partly ready."""
    def asset(name):
        return {"url": "https://github.com/Asher-1/ACloudViewer/releases/download/"
                       "main-devel/cloudviewer-3.9.5%2B19fae648-" + name,
                "size": "374.3 MB"}

    fixture = {
        "version_metadata": [
            {"value": version, "display_name": version, "is_default": version == "main-devel",
             "python_versions": ["3.10", "3.11", "3.12", "3.13"],
             "ubuntu_versions": ["ubuntu20.04", "ubuntu22.04", "ubuntu24.04"]}
            for version in ["main-devel", "v3.9.4", "linux-only"]],
        "download_data": {
            "main-devel": {
                # Exact production failure shape: Windows has wheels, no App.
                "windows": {"wheel": {
                    "cpu": {"amd64": {"3.10": {}, "3.11": {"url": ""},
                                       "3.12": asset("windows-cpu-cp312.whl"),
                                       "3.13": asset("windows-cpu-cp313.whl")},
                            "arm64": {"3.11": asset("windows-arm64-cp311.whl")}},
                    "cuda": {"amd64": {"3.11": asset("windows-cuda-cp311.whl")},
                             "arm64": {}}}},
                "macos": {"app": {}, "wheel": {"cpu": {"arm64": {}}}},
                "linux": {
                    "ubuntu20.04": {"app": {"cpu": {"amd64": asset("linux-app.run")}},
                                    "wheel": {}},
                    "ubuntu22.04": {"app": {}, "wheel": {
                        "cuda": {"amd64": {"3.11": asset("linux-cuda-cp311.whl")}}}},
                    "ubuntu24.04": {"app": {"cpu": {"amd64": {}}}, "wheel": {
                        "cpu": {"amd64": {"3.13": {"size": "0 MB"}}}}}}},
            "v3.9.4": {
                "windows": {"app": {"cpu": {"amd64": asset("stable-app.exe")},
                                    "cuda": {"amd64": {"size": "missing URL"}}},
                            "wheel": {"cpu": {"amd64": {"3.12": asset("stable-cp312.whl")},
                                              "arm64": {}}}},
                "macos": {"app": {"cpu": {"arm64": asset("stable-macos.dmg")}},
                          "wheel": {"cpu": {"arm64": {"3.11": asset("macos-cp311.whl")}}}},
                "linux": {"ubuntu20.04": {"app": {"cpu": {"amd64": asset("stable-linux.run")}}}}},
            "linux-only": {"windows": {}, "linux": {"ubuntu20.04": {
                "app": {"cpu": {"amd64": asset("linux-only.run")}}}}}},
    }

    def snapshot(page):
        return page.evaluate("""() => ({
            state: {...currentSelection},
            groups: Object.fromEntries(['os', 'linux-version', 'package', 'cuda', 'arch', 'python'].map(id =>
                [id, [...document.querySelectorAll('#' + id + '-selector .selector-btn')].map(b => ({
                    value: b.dataset.value, active: b.classList.contains('active'), disabled: b.disabled
                }))])),
            pythonVisible: getComputedStyle(document.querySelector('#python-row')).display !== 'none',
            linuxVisible: getComputedStyle(document.querySelector('#linux-version-row')).display !== 'none',
            href: document.querySelector('#download-output a.download-link')?.getAttribute('href'),
            unavailable: !!document.querySelector('#download-output .not-available')
        })""")

    def disabled(state, group, value):
        return any(row["value"] == value and row["disabled"]
                   for row in state["groups"][group])

    def click(page, group, value):
        page.locator(f'#{group}-selector .selector-btn[data-value="{value}"]').click()
        state = snapshot(page)
        # Every rendered active value must agree with the actual resolver state.
        fields = {"os": "os", "linux-version": "linuxVersion", "package": "package",
                  "cuda": "cuda", "arch": "arch", "python": "python"}
        rec.check(f"selection stays synchronized after {group}={value}", all(
            [row["value"] for row in state["groups"][name] if row["active"]] ==
            ([state["state"][field]] if state["state"][field] is not None else [])
            for name, field in fields.items()), str(state))
        return state

    page = context.new_page()
    page.set_default_timeout(5000)
    try:
        page.route("https://**/*", lambda route: route.abort())
        page.route("**/downloads_data.json*", lambda route: route.fulfill(json=fixture))
        page.goto(base + "/", wait_until="domcontentloaded")
        page.wait_for_selector('#version-selector .selector-btn.active')
        state = snapshot(page)
        rec.check("wheel-only Windows keeps beta and Windows, falls back to a real wheel",
                  state["state"] == {"version": "main-devel", "os": "windows", "package": "wheel",
                                     "linuxVersion": "ubuntu20.04", "python": "3.12",
                                     "cuda": "cpu", "arch": "amd64"} and state["pythonVisible"] and
                  state.get("href", "").endswith("windows-cpu-cp312.whl"), str(state))
        rec.check("unavailable App and empty OS/distro branches are disabled",
                  disabled(state, "package", "app") and disabled(state, "os", "macos") and
                  disabled(state, "linux-version", "ubuntu24.04"), str(state))
        rec.check("Python empty or URL-less leaves are not available",
                  disabled(state, "python", "3.10") and disabled(state, "python", "3.11"), str(state))
        state = click(page, "python", "3.13")
        rec.check("valid Python click changes the actual download leaf",
                  state["href"].endswith("windows-cpu-cp313.whl"), str(state))
        state = click(page, "arch", "arm64")
        rec.check("architecture change reconciles Python availability",
                  state["state"]["python"] == "3.11" and
                  state["href"].endswith("windows-arm64-cp311.whl"), str(state))
        state = click(page, "cuda", "cuda")
        rec.check("CUDA change rejects an empty architecture and updates Python",
                  state["state"]["arch"] == "amd64" and disabled(state, "arch", "arm64") and
                  state["state"]["python"] == "3.11" and
                  state["href"].endswith("windows-cuda-cp311.whl"), str(state))
        state = click(page, "os", "linux")
        rec.check("Linux App-only distro falls back from Wheel and clears Python",
                  state["state"]["package"] == "app" and state["state"]["python"] is None and
                  not state["pythonVisible"] and disabled(state, "package", "wheel") and
                  state["href"].endswith("linux-app.run"), str(state))
        state = click(page, "linux-version", "ubuntu22.04")
        rec.check("distro change selects its real Wheel CUDA branch",
                  state["state"]["package"] == "wheel" and state["state"]["cuda"] == "cuda" and
                  state["pythonVisible"] and disabled(state, "package", "app") and
                  disabled(state, "cuda", "cpu") and state["href"].endswith("linux-cuda-cp311.whl"), str(state))
        page.locator('#version-selector .selector-btn[data-value="v3.9.4"]').click()
        state = snapshot(page)
        rec.check("version change reconciles stale distro and package without switching OS",
                  state["state"]["os"] == "linux" and state["state"]["linuxVersion"] == "ubuntu20.04" and
                  state["state"]["package"] == "app" and state["href"].endswith("stable-linux.run"), str(state))
        state = click(page, "os", "windows")
        rec.check("available App remains preferred on an App-capable Windows version",
                  state["state"]["package"] == "app" and not state["pythonVisible"] and
                  disabled(state, "cuda", "cuda") and state["href"].endswith("stable-app.exe"), str(state))
        click(page, "package", "wheel")
        page.locator('#version-selector .selector-btn[data-value="main-devel"]').click()
        state = snapshot(page)
        rec.check("valid Wheel choice and Python survive a version change",
                  state["state"]["os"] == "windows" and state["state"]["package"] == "wheel" and
                  state["state"]["python"] == "3.12" and state["href"].endswith("windows-cpu-cp312.whl"), str(state))
        page.locator('#version-selector .selector-btn[data-value="v3.9.4"]').click()
        state = snapshot(page)
        rec.check("App reappearing does not discard a still-valid user Wheel choice",
                  state["state"]["package"] == "wheel" and state["href"].endswith("stable-cp312.whl"), str(state))
        state = click(page, "os", "macos")
        rec.check("macOS preserves valid Wheel choice and selects its CPU ARM Python leaf",
                  state["state"]["package"] == "wheel" and state["state"]["cuda"] == "cpu" and
                  state["state"]["arch"] == "arm64" and state["state"]["python"] == "3.11" and
                  state["href"].endswith("macos-cp311.whl") and
                  not page.locator('#cuda-row').is_visible(), str(state))
        state = click(page, "package", "app")
        rec.check("switching back to App clears Python and restores its download",
                  state["state"]["python"] is None and not state["pythonVisible"] and
                  state["href"].endswith("stable-macos.dmg"), str(state))
        state = click(page, "os", "windows")
        rec.check("returning to Windows keeps the valid App and reconciles architecture",
                  state["state"]["package"] == "app" and state["state"]["arch"] == "amd64" and
                  state["href"].endswith("stable-app.exe"), str(state))
        page.locator('#version-selector .selector-btn[data-value="linux-only"]').click()
        state = snapshot(page)
        rec.check("an entirely unavailable OS is disabled and a real OS remains selected",
                  state["state"]["os"] == "linux" and disabled(state, "os", "windows") and
                  state["href"].endswith("linux-only.run"), str(state))
    except Exception as error:
        rec.check("download dependency transitions complete", False, str(error))
    finally:
        page.close()

    page = context.new_page()
    try:
        empty = {"version_metadata": [fixture["version_metadata"][0]],
                 "download_data": {"main-devel": {"windows": {"app": {"cpu": {"amd64": {}}},
                                                                          "wheel": {}}}}}
        page.route("https://**/*", lambda route: route.abort())
        page.route("**/downloads_data.json*", lambda route: route.fulfill(json=empty))
        page.goto(base + "/", wait_until="domcontentloaded")
        page.wait_for_function('downloadsData !== null')
        state = snapshot(page)
        rec.check("no URL leaves keeps a genuine unavailable state with disabled choices",
                  state["unavailable"] and not state.get("href") and
                  all(row["disabled"] and not row["active"] for rows in state["groups"].values() for row in rows) and
                  page.locator('#version-selector .selector-btn').count() == 0, str(state))
    finally:
        page.close()


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
                logo_failures = []
                ctx.on("response", lambda response: logo_failures.append(response.url)
                       if response.status >= 400 and
                       "images/ACloudViewer_logo_horizontal.png" in response.url else None)

                def goto(path: str):
                    page = ctx.new_page()
                    page.goto(base + path, wait_until="domcontentloaded")
                    return page

                print("═══ live homepage navigation and download selectors ═══")
                home = ctx.new_page()
                home.route("https://**/*", lambda route: route.abort())
                response = home.goto(base + "/", wait_until="domcontentloaded")
                home.wait_for_selector("#download-output a.download-link", timeout=15000)
                rec.check("real homepage source renders with local CSS/JS",
                          response is not None and response.status == 200 and
                          home.title().startswith("ACloudViewer"), home.title())
                expected_nav = {"Home", "Features", "AICore AI", "Download", "Quick Start",
                                "Gallery", "Documentation", "Support", "GitHub"}
                actual_nav = {text.strip() for text in home.locator(
                    ".nav-menu > li > a").all_text_contents()}
                rec.check("all navigation links remain present",
                          expected_nav.issubset(actual_nav), str(sorted(actual_nav)))
                desktop = home.evaluate("""() => ({
                    width: innerWidth,
                    documentWidth: document.documentElement.scrollWidth,
                    navRight: document.querySelector('.nav-menu').getBoundingClientRect().right,
                    hamburger: getComputedStyle(document.querySelector('.hamburger')).display,
                    links: [...document.querySelectorAll('.nav-menu > li > a')].map(a => a.getBoundingClientRect().right)
                })""")
                rec.check("desktop navigation fits without hiding links at 1280px",
                          desktop["width"] == 1280 and desktop["documentWidth"] <= desktop["width"] and
                          desktop["navRight"] <= desktop["width"] and
                          all(x <= desktop["width"] for x in desktop["links"]), str(desktop))
                rec.check("desktop uses full navigation above its fit breakpoint",
                          desktop["hamburger"] == "none", desktop["hamburger"])
                for width in (1200, 1024, 768, 390, 320):
                    home.set_viewport_size({"width": width, "height": 844})
                    hamburger = home.locator(".hamburger")
                    rec.check(f"navigation collapses to hamburger at {width}px",
                              hamburger.is_visible() and
                              home.locator(".nav-menu").evaluate("e => !e.classList.contains('active')"))
                    hamburger.click()
                    home.wait_for_function("document.querySelector('.nav-menu').classList.contains('active')")
                    home.wait_for_function("getComputedStyle(document.querySelector('.nav-menu')).transform === 'matrix(1, 0, 0, 1, 0, 0)'")
                    opened = home.evaluate("""() => ({
                        width: innerWidth,
                        documentWidth: document.documentElement.scrollWidth,
                        bodyOverflow: getComputedStyle(document.body).overflowX,
                        menuRight: document.querySelector('.nav-menu').getBoundingClientRect().right,
                        links: document.querySelectorAll('.nav-menu > li > a').length
                    })""")
                    rec.check(f"whole homepage fits without document clipping at {width}px",
                              opened["documentWidth"] <= width and opened["bodyOverflow"] not in ("hidden", "clip"),
                              str(opened))
                    rec.check(f"all navigation links fit in expanded {width}px menu",
                              opened["links"] == len(expected_nav) and opened["menuRight"] <= width,
                              str(opened))
                    hamburger.click()

                code_layout = home.locator(".step-content .code-block").evaluate_all("""blocks => blocks.map(block => ({
                    right: block.getBoundingClientRect().right,
                    width: block.clientWidth,
                    contentWidth: block.scrollWidth,
                    overflow: getComputedStyle(block).overflowX,
                    command: block.querySelector('code').textContent,
                    copyButton: !!block.querySelector('.copy-btn')
                }))""")
                rec.check("long quickstart commands scroll within bounded code blocks",
                          bool(code_layout) and all(row["right"] <= 320 and row["overflow"] == "auto" and
                                                    row["copyButton"] for row in code_layout) and
                          any(row["contentWidth"] > row["width"] and "cv3d.__version__" in row["command"]
                              for row in code_layout), str(code_layout))

                empty_version = home.locator("#version-selector .selector-btn").filter(
                    has_text="ubuntu-prebuild-deps")
                rec.check("metadata with no downloadable files is not selectable",
                          empty_version.count() == 0)
                active_os = home.locator("#os-selector .selector-btn.active")
                rec.check("default Windows state is explicitly selected",
                          active_os.get_attribute("data-value") == "windows",
                          active_os.get_attribute("data-value"))
                default_download = home.locator("#download-output a.download-link")
                rec.check("default Windows choice matches its download target",
                          "main-devel-windows.exe" in default_download.get_attribute("href"),
                          default_download.get_attribute("href"))
                home.locator('#os-selector .selector-btn[data-value="linux"]').click()
                rec.check("changing OS synchronizes active state and options",
                          home.locator("#os-selector .selector-btn.active").get_attribute("data-value") == "linux" and
                          home.locator("#linux-version-row").is_visible() and
                          "main-devel-linux.run" in home.locator("#download-output a.download-link").get_attribute("href"))
                home.close()

                print("═══ actual download-leaf dependency transitions ═══")
                verify_download_availability(ctx, base, rec)

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

                print("═══ immutable legacy template compatibility ═══")
                legacy_errors = []
                legacy_page = ctx.new_page()
                legacy_page.on("pageerror", lambda error: legacy_errors.append(str(error)))
                for version in ("dev", "v3.9.4"):
                    legacy_page.goto(base + "/documentation/" + version + "/" + LEGACY_DEEP,
                                     wait_until="domcontentloaded")
                    legacy_page.wait_for_function("window.legacyVersionsLoaded === 1", timeout=5000)
                    rec.check("legacy versionsLoaded event delivered " + version,
                              legacy_page.evaluate("window.legacyVersionsLoaded") == 1)
                    for selector in ("#docs-version-select-sidebar", "#docs-version-select-main"):
                        opts = legacy_page.eval_on_selector_all(
                            selector + " option",
                            "els => els.map(e => [e.value, e.textContent.trim(), e.selected])")
                        rec.check("legacy selector reads manifest " + version + " " + selector,
                                  [o[0] for o in opts] == ids and
                                  any(o[0] == version and o[2] for o in opts) and
                                  all(o[1] and o[1] != "undefined" for o in opts), str(opts))
                    rec.check("legacy getVersions aliases preserve canonical fields " + version,
                              legacy_page.evaluate("""ACloudViewerVersionSwitcher.getVersions().every(
                                  v => v.value === v.id && v.display && v.label && v.kind)"""))
                # Exercise each immutable inline onchange. Also add the new marker
                # to one legacy select to prove a mixed template never navigates twice.
                for selector, target, mixed in (
                    ("#docs-version-select-sidebar", "dev", False),
                    ("#docs-version-select-main", "v3.9.4", True),
                ):
                    legacy_page.wait_for_function("window.legacyVersionsLoaded === 1")
                    if mixed:
                        legacy_page.locator(selector).evaluate("e => e.setAttribute('data-docs-version', '')")
                    probes = []
                    def record_probe(request):
                        if request.method == "HEAD":
                            probes.append(request.url)
                    legacy_page.on("request", record_probe)
                    legacy_page.select_option(selector, target)
                    legacy_page.wait_for_url("**/documentation/" + target + "/" + LEGACY_DEEP,
                                             timeout=8000)
                    legacy_page.wait_for_function("window.legacyVersionsLoaded === 1")
                    legacy_page.remove_listener("request", record_probe)
                    rec.check("legacy inline switch preserves deep page " + selector,
                              legacy_page.url.endswith("/documentation/" + target + "/" + LEGACY_DEEP))
                    rec.check("legacy inline switch sends one existence probe " + selector,
                              len(probes) == 1, str(probes))
                rec.check("immutable legacy templates have no JavaScript exceptions",
                          not legacy_errors, str(legacy_errors))
                # The aliases are runtime-only; deployment's manifest stays canonical.
                rec.check("legacy compatibility does not change versions.json schema",
                          all(set(v) == {"id", "label", "kind"} for v in manifest["versions"]))

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
                page.wait_for_function("document.getElementById('logo').naturalWidth > 0")
                rec.check("404 logo resolves from nested paths",
                          page.locator("#logo").evaluate("e => e.complete && e.naturalWidth > 0"))
                rec.check("404 logo never requests a broken relative URL",
                          not logo_failures, str(logo_failures))

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
