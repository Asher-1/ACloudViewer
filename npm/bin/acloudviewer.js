#!/usr/bin/env node
/**
 * @asher-1/acloudviewer — cross-platform installer bootstrap.
 *
 * Fetches the matching official ACloudViewer installer (.exe/.dmg/.run) or
 * Python wheel from the GitHub release and installs it. Zero dependencies:
 * Node >= 18 built-ins only (fetch, fs, child_process).
 *
 * Asset-matching rules mirror cli_anything/acloudviewer/utils/installer.py;
 * keep the two in sync when release asset names change.
 */

"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawn, spawnSync } = require("child_process");
const { Readable } = require("stream");

const GITHUB_REPO = "Asher-1/ACloudViewer";
const RELEASE_API = `https://api.github.com/repos/${GITHUB_REPO}/releases/tags/`;
const DEFAULT_RELEASE = "main-devel";
const USER_AGENT = "acloudviewer-npm-bootstrap";
// Same default install root as installer.py DEFAULT_APP_DIR: user-writable,
// no sudo needed for --silent installs.
const DEFAULT_TARGET = path.join(os.homedir(), ".local", "share", "ACloudViewer");

// ── Helpers ──────────────────────────────────────────────────────────────

function die(msg) {
  console.error(`error: ${msg}`);
  process.exit(1);
}

function isWindows() { return process.platform === "win32"; }
function isMacOS() { return process.platform === "darwin"; }
function isLinux() { return process.platform === "linux"; }

function archTag() {
  // Mirror installer.py PlatformInfo.arch_tag.
  return process.arch === "arm64" ? "ARM64" : "amd64";
}

function macArchTag() {
  // Mirror installer.py _match_macos_wheel tag preference.
  return process.arch === "arm64" ? "arm64" : "x86_64";
}

function detectNvidiaGpu() {
  const r = spawnSync("nvidia-smi", ["-L"], { timeout: 5000 });
  return r.status === 0;
}

function detectGlibcVersion() {
  if (!isLinux()) return "";
  const r = spawnSync("ldd", ["--version"], { encoding: "utf8" });
  if (r.status !== 0 || !r.stdout) return "";
  const m = r.stdout.split("\n")[0].match(/(\d+\.\d+)/);
  return m ? m[1] : "";
}

function detectLinuxDistro() {
  if (!isLinux()) return { id: "unknown", version: "" };
  try {
    const text = fs.readFileSync("/etc/os-release", "utf8");
    const info = {};
    for (const line of text.split("\n")) {
      const eq = line.indexOf("=");
      if (eq > 0) info[line.slice(0, eq).trim()] = line.slice(eq + 1).trim().replace(/^"|"$/g, "");
    }
    return { id: info.ID || "unknown", version: info.VERSION_ID || "" };
  } catch {
    return { id: "unknown", version: "" };
  }
}

function detectPythonInterpreter(explicit) {
  // Windows: try the py launcher first (always on PATH when the python.org
  // installer was used), then bare interpreters.
  const candidates = explicit ? [explicit]
    : (isWindows() ? ["py", "python", "python3"] : ["python3", "python"]);
  for (const exe of candidates) {
    const r = spawnSync(exe, ["--version"], { encoding: "utf8" });
    const out = `${r.stdout || ""}${r.stderr || ""}`;
    const m = out.match(/Python\s+(\d+)\.(\d+)/);
    if (r.status === 0 && m) {
      return { exe, tag: `cp${m[1]}${m[2]}`, version: `${m[1]}.${m[2]}` };
    }
  }
  return null;
}

function closestUbuntuTag(glibcVersion) {
  // Mirror installer.py _closest_ubuntu_tag.
  const map = { "2.27": "ubuntu18.04", "2.31": "ubuntu20.04", "2.35": "ubuntu22.04", "2.39": "ubuntu24.04" };
  if (map[glibcVersion]) return map[glibcVersion];
  const ver = parseFloat(glibcVersion);
  if (!Number.isFinite(ver)) return "ubuntu20.04";
  if (ver < 2.31) return "ubuntu18.04";
  if (ver < 2.35) return "ubuntu20.04";
  if (ver < 2.39) return "ubuntu22.04";
  return "ubuntu24.04";
}

// ── Release assets ───────────────────────────────────────────────────────

async function fetchReleaseAssets(tag) {
  // 30s hard timeout: a hung api.github.com call must not wedge the CLI.
  // GITHUB_TOKEN (optional) raises the anonymous 60/h per-IP limit to the
  // authenticated 5000/h — needed on shared CI runners and for agents doing
  // many release queries. Token is never logged.
  const headers = { "User-Agent": USER_AGENT, Accept: "application/vnd.github+json" };
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  let res;
  try {
    res = await fetch(RELEASE_API + tag, {
      headers,
      signal: AbortSignal.timeout(30000),
    });
  } catch (e) {
    throw new Error(`cannot reach GitHub API (${e && e.message ? e.message : e}). Check your network and try again.`);
  }
  if (res.status === 403) {
    throw new Error(`GitHub API refused the request for release "${tag}" (HTTP 403). ` +
      "Anonymous API calls are limited to 60/hour per IP — wait and retry, or set GITHUB_TOKEN to raise the limit, or check that the release exists at " +
      `https://github.com/${GITHUB_REPO}/releases`);
  }
  if (!res.ok) {
    throw new Error(`cannot read GitHub release "${tag}" (HTTP ${res.status}). Check https://github.com/${GITHUB_REPO}/releases`);
  }
  let release;
  try {
    release = await res.json();
  } catch {
    throw new Error(`unexpected non-JSON response from the GitHub API for "${tag}".`);
  }
  return release.assets || [];
}

function pickFlavor(args, hasNvidia) {
  // installer.py defaults to CUDA and falls back to CPU; --cpu/--cuda override.
  if (args.has("cpu")) return "cpu";
  if (args.has("cuda")) return "cuda";
  return hasNvidia ? "cuda" : "cpu";
}

function findInstallerApp(assets, plat, flavor) {
  // Mirror installer.py find_matching_app (desktop installers only).
  const isCpu = (n) => n.includes("-cpu-");
  const isCuda = (n) => n.includes("-cuda-");

  if (isLinux()) {
    const ubuntuTag = plat.distro.id === "ubuntu" && plat.distro.version
      ? `ubuntu${plat.distro.version}`
      : closestUbuntuTag(plat.glibc);
    let candidates = assets.filter((a) =>
      a.name.startsWith("ACloudViewer-") && a.name.endsWith(".run") &&
      a.name.includes(ubuntuTag) &&
      (flavor === "cpu" ? isCpu(a.name) : isCuda(a.name)));
    if (!candidates.length && flavor !== "cpu") {
      // CUDA missing -> CPU fallback (same as installer.py).
      candidates = assets.filter((a) =>
        a.name.startsWith("ACloudViewer-") && a.name.endsWith(".run") &&
        a.name.includes(ubuntuTag) && isCpu(a.name));
    }
    return candidates[0] || null;
  }
  if (isMacOS()) {
    // macOS ships CPU-only builds; the arch tag is the discriminator.
    return assets.find((a) =>
      a.name.startsWith("ACloudViewer-") && a.name.endsWith(".dmg") &&
      a.name.includes(plat.macArch)) || null;
  }
  if (isWindows()) {
    let candidates = assets.filter((a) =>
      a.name.startsWith("ACloudViewer-") && a.name.endsWith(".exe") &&
      (flavor === "cpu" ? isCpu(a.name) : isCuda(a.name)));
    if (!candidates.length && flavor !== "cpu") {
      candidates = assets.filter((a) =>
        a.name.startsWith("ACloudViewer-") && a.name.endsWith(".exe") && isCpu(a.name));
    }
    return candidates[0] || null;
  }
  return null;
}

function findWheel(assets, plat, py, cpuOnly) {
  // Mirror installer.py find_matching_wheel (+ _match_linux/_macos/_windows).
  const prefix = cpuOnly ? "cloudviewer_cpu-" : "cloudviewer-";
  const hasPyTag = (n) => n.includes(`-${py.tag}-${py.tag}-`);

  if (isLinux()) {
    const manylinux = plat.glibc
      ? `manylinux_${plat.glibc.replace(".", "_")}_${plat.macArch === "ARM64" ? "aarch64" : "x86_64"}`
      : "";
    let exact = assets.filter((a) =>
      a.name.startsWith(prefix) && hasPyTag(a.name) && manylinux && a.name.includes(manylinux));
    if (exact.length) return exact[0];
    const compatible = assets
      .filter((a) => a.name.startsWith(prefix) && hasPyTag(a.name) &&
        a.name.includes("manylinux") && a.name.includes("x86_64"))
      .sort((a, b) => a.name.localeCompare(b.name));
    return compatible[0] || null;
  }
  if (isMacOS()) {
    const matches = assets.filter((a) =>
      a.name.startsWith("cloudviewer-") && !a.name.startsWith("cloudviewer_cpu-") &&
      hasPyTag(a.name) && a.name.includes("macosx") && a.name.includes(plat.macArch));
    if (matches.length) return matches[0];
    const universal = assets.filter((a) =>
      a.name.startsWith("cloudviewer-") && hasPyTag(a.name) && a.name.includes("macosx"));
    return universal[0] || null;
  }
  if (isWindows()) {
    const matches = assets.filter((a) =>
      a.name.startsWith(prefix) && hasPyTag(a.name) && a.name.includes("win_amd64"));
    return matches[0] || null;
  }
  return null;
}

// ── Download ─────────────────────────────────────────────────────────────

function progressBar(current, total, speed) {
  const width = 30;
  const mb = (n) => (n / 1048576).toFixed(0);
  if (!total) return `  ${(current / 1048576).toFixed(1)} MB downloaded...`;
  const pct = Math.min(Math.floor((current * 100) / total), 100);
  const filled = Math.floor((width * current) / total);
  const bar = "█".repeat(filled) + "░".repeat(width - filled);
  const rate = speed !== undefined ? `  ${speed.toFixed(1)} MB/s` : "";
  return `  [${bar}] ${pct}% ${mb(current)}/${mb(total)} MB${rate}`;
}

// Aggregated progress over one or many concurrent byte streams. TTY: redraw
// a \r line at most every 200ms (with speed). Non-TTY (CI logs, pipes): print
// one line per ~10% (or 10 MB) — \r spam would flood the log.
function makeProgress(total) {
  const isTTY = process.stdout.isTTY;
  let got = 0, lastPrint = 0, lastGot = 0, lastTime = Date.now();
  const print = (now) => {
    const dt = (now - lastTime) / 1000;
    const speed = dt > 0 ? (got - lastGot) / dt : undefined;
    lastGot = got;
    lastTime = now;
    process.stdout.write(`\r${progressBar(got, total, speed)}`);
  };
  return {
    add(n) {
      got += n;
      const now = Date.now();
      if (isTTY) {
        if (now - lastPrint >= 200) { lastPrint = now; print(now); }
      } else {
        const step = total ? total / 10 : 10 * 1048576;
        if (got - lastGot >= step) {
          lastGot = got;
          lastPrint = now;
          console.log(progressBar(got, total));
        }
      }
    },
    finish() {
      if (isTTY) process.stdout.write("\n");
      else console.log(progressBar(got, total));
    },
  };
}

// ── Download engine ──────────────────────────────────────────────────────
//
// Zero-dependency downloader with three layers, cheapest first:
//   1. probe    — a Range: bytes=0-0 request discovers size + range support
//   2. segmented— N parallel Range requests written at absolute offsets
//                 (helps when the bottleneck is per-connection throttling;
//                 measured 2.4x aggregate on release CDN assets)
//   3. single   — one stream with idle-stall abort + Range resume on retry
// Every response that proves ranges do not work (200 to a Range request,
// content-encoding, garbage content-range) degrades to the single stream.

const SEGMENT_THRESHOLD = 64 * 1048576;  // only segment files >= 64 MB
const MAX_PARALLEL = 16;
const STALL_TIMEOUT_MS = 60000;          // abort after 60s without any bytes
const SEGMENT_RETRIES = 8;               // with 60s backoff cap: survives a
                                         // ~4 min network outage mid-download

class RangesUnsupportedError extends Error {
  constructor() { super("server does not honor range requests"); }
}

class AssetGoneError extends Error {
  constructor(status) {
    super(`asset is no longer downloadable (HTTP ${status}) — the main-devel ` +
      "release is updated between builds and the asset was replaced or removed " +
      "during this download; re-run to fetch the current assets");
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Human-readable failure reason: fetch wraps low-level errors as
// "fetch failed" and keeps the real cause (ECONNRESET, getaddrinfo ...)
// in err.cause — surface it, a bare "fetch failed" is undebuggable.
function errorReason(e) {
  if (e && e.name === "AbortError") return `stalled (${STALL_TIMEOUT_MS / 1000}s without data)`;
  const cause = e && e.cause && (e.cause.code || e.cause.message)
    ? ` (${e.cause.code || e.cause.message})` : "";
  return `${(e && e.message) || String(e)}${cause}`;
}

// Range probe: 206 + content-range => size + ranges OK; 200 => whole-file
// reply, ranges unsupported. The body is always cancelled: in the 200 case
// it IS the entire file, and letting it stream would duplicate the transfer.
async function probeRange(url) {
  const res = await fetch(url, {
    headers: { "User-Agent": USER_AGENT, Range: "bytes=0-0" },
    redirect: "follow",
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) {
    if (res.status === 404 || res.status === 410) throw new AssetGoneError(res.status);
    throw new Error(`HTTP ${res.status} while probing ${url}`);
  }
  try {
    if (res.status !== 206) {
      return { ranges: false, total: Number(res.headers.get("content-length") || 0) };
    }
    if (res.headers.get("content-encoding")) return { ranges: false, total: 0 };
    const total = Number(((res.headers.get("content-range") || "").split("/")[1] || "").trim());
    return Number.isFinite(total) && total > 0
      ? { ranges: true, total }
      : { ranges: false, total: 0 };
  } finally {
    if (res.body) await res.body.cancel().catch(() => {});
  }
}

// N parallel Range requests into one preallocated file. Each worker writes
// its own byte range at absolute offsets via positional fs.writeSync on the
// shared fd — writeSync is atomic within the event loop and workers never
// overlap offsets, so no locking is needed. Memory: MAX_PARALLEL workers x
// stream chunk (~64KB) per segment, i.e. a few MB total, never the file size.
// why not O(file): total work is O(bytes) but wall time drops ~N-fold under
// per-connection throttling (measured 2.4x with N=4 on release CDN assets).
async function downloadSegmented(url, dest, total, segments, progress) {
  const fd = fs.openSync(dest, "w");
  fs.ftruncateSync(fd, total);
  const segSize = Math.ceil(total / segments);
  const segs = [];
  for (let i = 0; i < segments; i++) {
    const start = i * segSize;
    segs.push({ start, end: Math.min(start + segSize, total) - 1, done: 0, ok: false });
  }
  const state = { cancelled: false };
  const results = await Promise.allSettled(segs.map(async (seg, idx) => {
    for (let attempt = 1; attempt <= SEGMENT_RETRIES && !seg.ok; attempt++) {
      const ctrl = new AbortController();
      let timer = setTimeout(() => ctrl.abort(), STALL_TIMEOUT_MS);
      try {
        const res = await fetch(url, {
          headers: { "User-Agent": USER_AGENT, Range: `bytes=${seg.start + seg.done}-${seg.end}` },
          redirect: "follow",
          signal: ctrl.signal,
        });
        if (res.status === 200) throw new RangesUnsupportedError();
        if (res.status === 404 || res.status === 410) {
          // main-devel is a rolling release: update-release deletes and
          // re-uploads assets between builds, so an in-flight download can
          // legitimately outlive its asset (verified 2026-11-15: segment 404
          // after the release was refreshed mid-download). Retrying cannot
          // help — fail fast with an actionable message.
          throw new AssetGoneError(res.status);
        }
        if (res.status !== 206) throw new Error(`HTTP ${res.status} for segment range`);
        if (res.headers.get("content-encoding")) throw new RangesUnsupportedError();
        for await (const chunk of Readable.fromWeb(res.body)) {
          if (state.cancelled) throw new Error("cancelled");
          fs.writeSync(fd, chunk, 0, chunk.length, seg.start + seg.done);
          seg.done += chunk.length;
          progress.add(chunk.length);
          timer.refresh();  // idle timer resets on every received chunk
        }
        clearTimeout(timer);
        seg.ok = true;
      } catch (e) {
        clearTimeout(timer);
        if (e instanceof RangesUnsupportedError) { state.cancelled = true; throw e; }
        if (e instanceof AssetGoneError) { state.cancelled = true; throw e; }
        if (attempt < SEGMENT_RETRIES) {
          const wait = Math.min(2 * 2 ** (attempt - 1), 60);
          console.log(`  segment ${idx + 1}/${segments}: retry ${attempt}/${SEGMENT_RETRIES - 1} in ${wait}s (${errorReason(e)})`);
          await sleep(wait * 1000);
        } else {
          throw new Error(`segment ${idx + 1}/${segments} (bytes ${seg.start}-${seg.end}) failed after ${SEGMENT_RETRIES} attempts: ${errorReason(e)}`);
        }
      }
    }
  }));
  fs.closeSync(fd);
  const unsupported = results.find((r) => r.status === "rejected" && r.reason instanceof RangesUnsupportedError);
  if (unsupported) throw unsupported.reason;
  const gone = results.find((r) => r.status === "rejected" && r.reason instanceof AssetGoneError);
  if (gone) throw gone.reason;
  const failed = results.filter((r) => r.status === "rejected").map((r) => r.reason && r.reason.message);
  if (failed.length) throw new Error(`${failed.length}/${segments} download segment(s) failed:\n  - ${failed.join("\n  - ")}`);
  const size = fs.statSync(dest).size;
  if (size !== total) throw new Error(`downloaded size mismatch: ${size}/${total} bytes`);
}

// Single stream with idle-stall abort and Range resume: on retry the next
// attempt asks for bytes=<written>- instead of restarting from zero, which
// matters at low bandwidth (a 278 MB transfer that drops at 90% must not
// restart). A 200 reply to a resume request means the server ignored the
// Range header -> truncate and restart cleanly.
async function downloadSingleStream(url, dest, info, progress, retries = 8) {
  let written = fs.existsSync(dest) ? fs.statSync(dest).size : 0;
  if (written > info.total) written = 0;  // stale/larger file: start over
  for (let attempt = 1; attempt <= retries; attempt++) {
    let fd = null;
    const ctrl = new AbortController();
    let timer = setTimeout(() => ctrl.abort(), STALL_TIMEOUT_MS);
    try {
      const headers = { "User-Agent": USER_AGENT };
      if (written > 0 && info.ranges) headers.Range = `bytes=${written}-`;
      const res = await fetch(url, { headers, redirect: "follow", signal: ctrl.signal });
      if (res.status === 404 || res.status === 410) throw new AssetGoneError(res.status);
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      if (res.headers.get("content-encoding")) {
        throw new Error("unexpected content-encoding on asset download; refusing to write a corrupt file");
      }
      if (res.status === 200 && written > 0) {
        written = 0;  // server ignored the resume range
      }
      if (info.total && written === info.total) return;  // already complete
      fd = fs.openSync(dest, written > 0 ? "r+" : "w");
      if (written > 0) fs.ftruncateSync(fd, written);
      for await (const chunk of Readable.fromWeb(res.body)) {
        fs.writeSync(fd, chunk, 0, chunk.length, written);
        written += chunk.length;
        progress.add(chunk.length);
        timer.refresh();
      }
      clearTimeout(timer);
      if (info.total && written < info.total) {
        throw new Error(`connection closed early: ${written}/${info.total} bytes`);
      }
      return;
    } catch (e) {
      clearTimeout(timer);
      if (e instanceof AssetGoneError) throw e;
      if (attempt < retries) {
        const wait = Math.min(5 * 2 ** (attempt - 1), 60);
        console.log(`  retry ${attempt}/${retries - 1} in ${wait}s (${errorReason(e)})`);
        await sleep(wait * 1000);
      } else {
        throw new Error(`download failed after ${retries} attempts: ${errorReason(e)}`);
      }
    } finally {
      if (fd !== null) { try { fs.closeSync(fd); } catch { /* already closed */ } }
    }
  }
}

async function downloadWithProgress(url, dest, label, opts = {}) {
  const probe = await probeRange(url);
  const total = probe.total || 0;
  // Segments: explicit --parallel N wins; otherwise auto (4) for large files
  // on range-capable servers. Everything unsupported degrades to 1.
  let segments = opts.parallel || (probe.ranges && total >= SEGMENT_THRESHOLD ? 4 : 1);
  segments = Math.max(1, Math.min(MAX_PARALLEL, Math.floor(segments) || 1));
  if (segments > 1 && (!probe.ranges || (total && total < segments))) {
    console.log("  note: server does not support range requests; single-stream download.");
    segments = 1;
  }
  const progress = makeProgress(total);
  console.log(`Downloading ${label}${total ? ` (${(total / 1048576).toFixed(0)} MB${segments > 1 ? `, ${segments} parallel segments` : ""})` : ""} ...`);
  if (segments > 1) {
    try {
      await downloadSegmented(url, dest, total, segments, progress);
      progress.finish();
      return dest;
    } catch (e) {
      if (e instanceof RangesUnsupportedError) {
        console.log("  note: server did not honor range requests; falling back to a single stream.");
        if (fs.existsSync(dest)) fs.unlinkSync(dest);
      } else {
        throw e;
      }
    }
  }
  await downloadSingleStream(url, dest, { total, ranges: probe.ranges }, progress);
  progress.finish();
  return dest;
}

// ── Install actions ──────────────────────────────────────────────────────

function findBinaryInDir(target) {
  // Real QtIFW layout (verified end-to-end on Linux): the binary and its
  // .sh/.bat launcher sit at the target root, no bin/ subdir.
  const name = isWindows() ? "ACloudViewer.exe" : "ACloudViewer";
  for (const p of [path.join(target, name), path.join(target, "bin", name)]) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function runInstaller(asset, silent, target) {
  if (isMacOS() && asset.name.endsWith(".dmg")) {
    if (silent) return installDmgSilent(asset, target);
    // Attach synchronously BEFORE returning: the tmp dir holding the dmg is
    // removed right after this call, so the volume must exist by then.
    console.log("Mounting disk image ...");
    const r = spawnSync("hdiutil", ["attach", "-nobrowse", "-readonly", asset.path], { encoding: "utf8" });
    if (r.status !== 0) throw new Error(`hdiutil attach failed: ${(r.stderr || "").slice(0, 300)}`);
    let volume = null;
    for (const line of (r.stdout || "").trim().split("\n")) {
      const parts = line.split("\t");
      if (parts.length >= 3) volume = parts[parts.length - 1].trim();
    }
    if (!volume || !fs.existsSync(volume)) throw new Error("DMG mounted but volume not found.");
    spawn("open", [volume], { detached: true, stdio: "ignore" }).unref();
    console.log(`\nMounted ${volume} — drag ACloudViewer.app to Applications in the Finder window.`);
    return;
  }
  if (silent) {
    // Mirror installer.py's production-verified QtIFW console invocations:
    //   Linux .run : in --accept-licenses --confirm-command --root <dir>
    //   Win/mac    : --accept-licenses --accept-messages --confirm-command install --root <dir>
    // All land in --root (user-writable by default) instead of /opt.
    const args = isWindows()
      ? ["--root", target, "--accept-licenses", "--accept-messages", "--confirm-command", "install"]
      : ["in", "--accept-licenses", "--confirm-command", "--root", target];
    console.log(`Running silent install (QtIFW console mode) -> ${target}`);
    const env = { ...process.env };
    if (!isWindows()) env.QT_QPA_PLATFORM = env.QT_QPA_PLATFORM || "minimal";
    // QtIFW refuses to install over an existing target (TargetDirectoryInUse).
    // Mirror installer.py's upgrade semantics: remove the previous
    // installation through its own maintenance tool (never a raw rm).
    const mt = path.join(target, isWindows() ? "maintenancetool.exe" : "maintenancetool");
    if (fs.existsSync(mt)) {
      console.log("Existing installation found; uninstalling via maintenancetool purge ...");
      const p = spawnSync(mt, ["purge", "--confirm-command"], { stdio: "inherit", env });
      if (p.status !== 0) {
        throw new Error(`maintenancetool purge exited with code ${p.status}; remove ${target} manually or choose another --prefix.`);
      }
    }
    fs.mkdirSync(target, { recursive: true });
    const r = spawnSync(asset.path, args, { stdio: "inherit", env });
    if (r.status !== 0) {
      throw new Error(`installer exited with code ${r.status}; retry without --silent for the GUI wizard, or with --prefix <dir> for a different install root.`);
    }
    const binary = findBinaryInDir(target);
    if (binary) console.log(`\nInstalled. Binary: ${binary}`);
    else console.log(`\nInstalled to ${target} (binary not auto-detected; check contents).`);
    return;
  }
  if (isLinux() && !process.env.DISPLAY && !process.env.WAYLAND_DISPLAY) {
    console.error("warning: no graphical session detected; the QtIFW wizard may not open. Use --silent for a headless install.");
  }
  const child = spawn(asset.path, [], { detached: true, stdio: "ignore" });
  child.on("error", (e) =>
    console.error(`warning: installer launch failed (${e.message}); download again with 'acloudviewer install' and run it manually.`));
  child.unref();
  console.log(`\nInstaller window opened: ${asset.name}`);
  console.log("Follow the wizard to finish. (Use --silent for unattended install.)");
}

function installDmgSilent(asset, target) {
  // Mirror installer.py _install_macos_dmg: attach -> run the bundled QtIFW
  // installer headlessly with --root -> detach. Throws (never process.exit)
  // so the tmp-dir cleanup in cmdInstall's finally still runs.
  console.log("Attaching disk image ...");
  const r = spawnSync("hdiutil", ["attach", "-nobrowse", "-readonly", asset.path], { encoding: "utf8" });
  if (r.status !== 0) throw new Error(`hdiutil attach failed: ${(r.stderr || "").slice(0, 300)}`);
  let volume = null;
  for (const line of (r.stdout || "").trim().split("\n")) {
    const parts = line.split("\t");
    if (parts.length >= 3) volume = parts[parts.length - 1].trim();
  }
  if (!volume || !fs.existsSync(volume)) throw new Error("DMG mounted but volume not found.");
  try {
    const apps = fs.readdirSync(volume).filter((n) => n.endsWith(".app"));
    if (!apps.length) throw new Error(`no .app bundle inside ${volume}.`);
    const macosDir = path.join(volume, apps[0], "Contents", "MacOS");
    const installerBin = fs.readdirSync(macosDir).find((n) => {
      try { fs.accessSync(path.join(macosDir, n), fs.constants.X_OK); return true; } catch { return false; }
    });
    if (!installerBin) throw new Error(`no executable installer inside ${apps[0]}.`);
    fs.mkdirSync(target, { recursive: true });
    console.log(`Installing to ${target} (QtIFW headless) ...`);
    const env = { ...process.env, QT_QPA_PLATFORM: process.env.QT_QPA_PLATFORM || "minimal" };
    const run = spawnSync(path.join(macosDir, installerBin),
      ["--root", target, "--accept-licenses", "--accept-messages", "--confirm-command", "install"],
      { stdio: "inherit", env });
    if (run.status !== 0) throw new Error(`installer exited with code ${run.status}. Try opening ${asset.name} manually.`);
  } finally {
    spawnSync("hdiutil", ["detach", volume, "-quiet"]);
  }
  console.log(`Installed. Launch from ${target}.`);
}

function installWheelFile(asset, pythonExe) {
  console.log("Installing wheel with pip ...");
  const r = spawnSync(pythonExe, ["-m", "pip", "install", asset.path], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`pip install exited with code ${r.status}.`);
  console.log("Installed. Try: python -c \"import cloudViewer; print(cloudViewer.__version__)\"");
}

// ── Commands ─────────────────────────────────────────────────────────────

function parseParallel(args) {
  const v = args.get("parallel");
  if (v === undefined) return undefined;
  if (typeof v !== "string" || !/^\d+$/.test(v)) {
    throw new Error("--parallel requires a number between 1 and 16 (e.g. --parallel 4).");
  }
  const n = Number(v);
  if (n < 1 || n > MAX_PARALLEL) {
    throw new Error(`--parallel requires a number between 1 and ${MAX_PARALLEL}.`);
  }
  return n;
}

async function cmdInstall(args) {
  const tag = args.get("release") || DEFAULT_RELEASE;
  const silent = args.has("silent");
  const wheel = args.has("wheel");
  const cpuOnly = args.has("cpu") || args.has("cpu-only");
  const parallel = parseParallel(args);

  // Local-file install (mirrors installer.py install_app_from_file): skips
  // the release query and download entirely. Useful for offline installs
  // and for testing the silent path with an already-downloaded installer.
  const from = args.get("from");
  if (from !== undefined && wheel) {
    throw new Error("--from and --wheel cannot be combined.");
  }
  if (from !== undefined) {
    if (typeof from !== "string" || !from) throw new Error("--from requires a file path argument.");
    if (!fs.existsSync(from)) throw new Error(`file not found: ${from}`);
    const prefixArg = args.get("prefix");
    if (prefixArg && typeof prefixArg !== "string") throw new Error("--prefix requires a directory argument.");
    const asset = { name: path.basename(from), path: from, size: fs.statSync(from).size };
    const target = prefixArg || DEFAULT_TARGET;
    if (args.has("dry-run")) {
      // Zero side effects: no chmod, no installer execution (verified gap:
      // --from previously ignored --dry-run and really launched the file).
      console.log(`would install from local file: ${asset.name} (${(asset.size / 1048576).toFixed(0)} MB) -> ${target}` +
        (silent ? " (silent QtIFW console mode)" : " (GUI wizard)"));
      console.log("dry-run: skipping install.");
      return;
    }
    console.log(`Installing from local file: ${asset.name} (${(asset.size / 1048576).toFixed(0)} MB)`);
    if (isLinux()) fs.chmodSync(from, 0o755);
    runInstaller(asset, silent, target);
    return;
  }

  const assets = await fetchReleaseAssets(tag);

  if (wheel) {
    const py = detectPythonInterpreter(args.get("python"));
    if (!py) die("no Python interpreter found; install Python 3.10+ or pass --python <path>.");
    console.log(`Platform: ${process.platform}/${process.arch}  Python: ${py.version} (${py.exe})`);
    const asset = findWheel(assets, platformInfo(args), py, cpuOnly);
    if (!asset) die(`no matching wheel for ${py.version} on ${process.platform}/${process.arch}. Run 'acloudviewer list' to see available assets.`);
    console.log(`Matched wheel: ${asset.name} (${(asset.size / 1048576).toFixed(0)} MB)`);
    if (args.has("dry-run")) {
      console.log(`Download URL: ${asset.browser_download_url}`);
      console.log("dry-run: skipping download and pip install.");
      return;
    }
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "acv-wheel-"));
    try {
      asset.path = await downloadWithProgress(asset.browser_download_url, path.join(tmp, asset.name), asset.name, { parallel });
      installWheelFile(asset, py.exe);
    } finally {
      try {
        fs.rmSync(tmp, { recursive: true, force: true });
      } catch { /* best effort */ }
    }
    return;
  }

  const plat = platformInfo(args);
  console.log(`Platform: ${process.platform}/${process.arch}` +
    (isLinux() ? `  distro: ${plat.distro.id} ${plat.distro.version}  glibc: ${plat.glibc}` : "") +
    (plat.hasNvidia ? "  GPU: NVIDIA detected" : ""));
  const flavor = pickFlavor(args, plat.hasNvidia);
  if (isMacOS() && flavor === "cuda") {
    console.log("note: macOS has no CUDA build; using the CPU installer.");
  }
  const asset = findInstallerApp(assets, plat, isMacOS() ? "cpu" : flavor);
  if (!asset) die(`no matching installer on ${process.platform}/${process.arch} (flavor: ${flavor}). Run 'acloudviewer list' to see available assets.`);
  console.log(`Matched installer: ${asset.name} (${(asset.size / 1048576).toFixed(0)} MB)`);
  if (args.has("dry-run")) {
    console.log(`Download URL: ${asset.browser_download_url}`);
    console.log("dry-run: skipping download and install.");
    return;
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "acv-install-"));
  try {
    asset.path = await downloadWithProgress(asset.browser_download_url, path.join(tmp, asset.name), asset.name, { parallel });
    if (isLinux()) fs.chmodSync(asset.path, 0o755);
    const prefixArg = args.get("prefix");
    if (prefixArg && typeof prefixArg !== "string") {
      throw new Error("--prefix requires a directory argument.");
    }
    runInstaller(asset, silent, prefixArg || DEFAULT_TARGET);
  } finally {
    // Best-effort cleanup: on Windows a GUI-launched .exe locks its file, so
    // removal may fail — keep the dir and tell the user instead of crashing.
    try {
      fs.rmSync(tmp, { recursive: true, force: true });
    } catch (e) {
      console.log(`note: could not fully clean temp dir ${tmp} (${e.message}); it can be deleted manually.`);
    }
  }
}

function platformInfo(args) {
  const distro = detectLinuxDistro();
  const glibc = detectGlibcVersion();
  return {
    arch: archTag(),
    macArch: macArchTag(),
    glibc,
    distro,
    hasNvidia: args.has("cuda") ? true : (args.has("cpu") ? false : detectNvidiaGpu()),
  };
}

function mb(bytes) { return (bytes / 1048576).toFixed(0); }

async function cmdList(args) {
  const tag = args.get("release") || DEFAULT_RELEASE;
  const assets = await fetchReleaseAssets(tag);
  console.log(`Release "${tag}" on ${GITHUB_REPO}:`);
  const apps = assets.filter((a) => a.name.startsWith("ACloudViewer-"));
  const wheels = assets.filter((a) => a.name.endsWith(".whl"));
  if (apps.length) {
    console.log("\nDesktop installers:");
    for (const a of apps) console.log(`  ${a.name}  ${mb(a.size)} MB`);
  }
  if (wheels.length) {
    console.log("\nPython wheels:");
    for (const a of wheels) console.log(`  ${a.name}  ${mb(a.size)} MB`);
  }
}

async function cmdUninstall(args) {
  // NOT a mirror of installer.py (which has no uninstall command): this
  // wraps the QtIFW maintenance tool's own purge, whose semantics were
  // verified end-to-end (removes the install dir contents, exit 0).
  const prefixArg = args.get("prefix");
  if (prefixArg && typeof prefixArg !== "string") throw new Error("--prefix requires a directory argument.");
  const target = prefixArg || DEFAULT_TARGET;
  const mtName = isWindows() ? "maintenancetool.exe" : "maintenancetool";
  // Verified real QtIFW layout: maintenancetool sits at the target root;
  // check bin/ too in case a future repack moves it.
  const mt = [path.join(target, mtName), path.join(target, "bin", mtName)]
    .find((p) => fs.existsSync(p));
  if (!mt) {
    throw new Error(
      `no QtIFW maintenance tool found under ${target}. ` +
      "If ACloudViewer is installed in another location, pass --prefix <dir>; " +
      "if it is not installed, there is nothing to uninstall.");
  }
  console.log(`ACloudViewer installation found at: ${target}`);
  console.log(`Uninstaller: ${mt} (purge removes ALL files the installer placed in that directory)`);
  if (!args.has("yes")) {
    // Safety gate: purge is destructive, so a non-interactive shell (agents,
    // CI, piped input) must opt in explicitly instead of being surprised.
    if (!process.stdin.isTTY) {
      throw new Error("refusing to uninstall without confirmation in a non-interactive shell; re-run with --yes.");
    }
    const rl = require("readline").createInterface({ input: process.stdin, output: process.stdout });
    const answer = (await new Promise((resolve) => rl.question("Proceed with uninstall? [y/N] ", resolve))).trim().toLowerCase();
    rl.close();
    if (answer !== "y" && answer !== "yes") {
      console.log("aborted.");
      return;
    }
  }
  const env = { ...process.env };
  if (!isWindows()) env.QT_QPA_PLATFORM = env.QT_QPA_PLATFORM || "minimal";
  const r = spawnSync(mt, ["purge", "--confirm-command"], { stdio: "inherit", env });
  if (r.status !== 0) {
    throw new Error(`maintenancetool purge exited with code ${r.status}.`);
  }
  const leftover = fs.existsSync(target) ? fs.readdirSync(target).length : 0;
  console.log(leftover === 0
    ? `\nUninstalled. ${target} is gone.`
    : `\nUninstall finished; ${leftover} item(s) remain under ${target} (installer logs or shortcuts the purger does not own).`);
}

function usage() {
  console.log(`ACloudViewer installer bootstrap (npm package @asher-1/acloudviewer)

Usage:
  acloudviewer install [--cuda|--cpu] [--silent] [--prefix <dir>] [--parallel <n>] [--release <tag>]
  acloudviewer install --from <installer-file> [--silent] [--prefix <dir>]
  acloudviewer install --wheel [--cpu-only] [--python <path>] [--release <tag>]
  acloudviewer list [--release <tag>]
  acloudviewer uninstall [--prefix <dir>] [--yes]
  acloudviewer version | help

Options:
  --cuda / --cpu      Force a build flavor (default: auto-detect NVIDIA GPU).
  --silent            Unattended install (QtIFW console mode / dmg headless).
  --prefix <dir>      Install root for --silent (default:
                      ~/.local/share/ACloudViewer, same as the pip harness).
  --parallel <n>      Concurrent download segments, 1-${MAX_PARALLEL} (default: 4 for
                      files >= 64 MB when the server supports ranges; falls
                      back to a single stream automatically).
  --wheel             Install the Python wheel instead of the desktop app
                      (requires Python 3.10+).
  --from <file>       Install from a local installer file (.exe/.dmg/.run)
                      instead of downloading from GitHub Releases.
  --yes               For uninstall: skip the confirmation prompt.
  --dry-run           Print the matched asset and URL, then exit.
  --release <tag>     GitHub release tag (default: ${DEFAULT_RELEASE}).

Downloads land in a temp dir and are removed after install. Segmented
downloads write into one preallocated file and abort any stream that stalls
for ${STALL_TIMEOUT_MS / 1000}s; failed attempts resume from the bytes already
received instead of restarting. Uninstall runs the QtIFW maintenancetool
("purge") of the installation directory — the same tool the install wizard
provides.`);
}

// ── Main ─────────────────────────────────────────────────────────────────

async function main() {
  const argv = process.argv.slice(2);
  const cmd = argv[0] || "help";
  const args = new Map();
  for (let i = 1; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) { args.set(argv[i].slice(2), next); i++; }
      else args.set(argv[i].slice(2), true);
    }
  }
  switch (cmd) {
    case "install": await cmdInstall(args); break;
    case "list": await cmdList(args); break;
    case "uninstall": await cmdUninstall(args); break;
    case "version": console.log(require("../package.json").version); break;
    case "help": usage(); break;
    default: console.error(`unknown command: ${cmd}\n`); usage(); process.exit(1);
  }
}

// Expose the download engine for tests when require()d as a module; run the
// CLI only when executed directly.
if (require.main === module) {
  module.exports = { downloadWithProgress };
  main().catch((err) => die(err && err.message ? err.message : String(err)));
} else {
  module.exports = { downloadWithProgress };
}
