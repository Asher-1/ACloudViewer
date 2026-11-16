# @asher-1/acloudviewer

Cross-platform **installer bootstrap** for [ACloudViewer](https://asher-1.github.io/ACloudViewer/) — an open-source 3D point cloud and mesh processing application.

The npm package itself is tiny (~KB): it detects your platform, downloads the matching official installer (`.exe` / `.dmg` / `.run`) **or** Python wheel from [GitHub Releases](https://github.com/Asher-1/ACloudViewer/releases), and launches it. The binary distribution always comes from GitHub Releases — npm never ships the large binaries.

## Install

### Channel 1: npmjs.org (no authentication, recommended)

```bash
npm install -g @asher-1/acloudviewer
acloudviewer install        # downloads + launches the installer for your platform
```

### Channel 2: GitHub Packages (mirror)

GitHub Packages requires a token to install **any** package, including public ones. One-time setup:

```bash
# 1. Create a PAT (classic) with read:packages at
#    https://github.com/settings/tokens/new?scopes=read:packages
# 2. Configure npm once:
npm config set @asher-1:registry https://npm.pkg.github.com/
npm config set //npm.pkg.github.com/:_authToken=<YOUR_TOKEN>
# 3. Install:
npm install -g @asher-1/acloudviewer
acloudviewer install
```

No Node.js on the target machine? Any user can also download installers directly at https://github.com/Asher-1/ACloudViewer/releases.

## Usage

```text
acloudviewer install [--cuda] [--force-cuda] [--silent] [--prefix <dir>] [--parallel <n>] [--dry-run] [--release <tag>]
acloudviewer install --wheel [--cuda] [--force-cuda] [--python <path>] [--release <tag>]
acloudviewer list [--release <tag>]
acloudviewer uninstall [--prefix <dir>] [--yes]
acloudviewer version | help
```

| Option | Meaning |
|--------|---------|
| `--cuda` | Opt into the CUDA build (desktop or wheel). **Default: CPU** — GPU presence never auto-selects CUDA. Before any download the NVIDIA driver is checked: it must support CUDA **12.x** (minor versions within a major are compatible; only a lower major, e.g. an 11.x driver, is incompatible). An incompatible host is offered: re-check after a driver upgrade / abort / install the CPU build; on a **major mismatch** additionally: build from source via the CLI harness (`cli-anything-acloudviewer install source`, installed via pip automatically if missing). Non-interactive shells refuse with guidance instead of prompting. |
| `--force-cuda` | Bypass the CUDA driver check (for setups where `nvidia-smi` misreports; the app may fail at runtime). |
| `--cpu` / `--cpu-only` | Accepted for compatibility; CPU is already the default. |
| `--silent` | Unattended install (QtIFW console mode; the app lands in `--prefix`, default `~/.local/share/ACloudViewer` — no sudo). |
| `--prefix <dir>` | Install root for `--silent` / `--from`, and the target for `uninstall`. |
| `--parallel <n>` | Concurrent download segments, 1–16 (see *Download behavior*). |
| `--wheel` | Install the Python `cloudViewer` wheel via pip instead of the desktop app (requires Python 3.10+). |
| `--dry-run` | Print the matched asset and download URL, then exit (no download). |
| `--release <tag>` | GitHub release tag; defaults to the rolling `main-devel` release. |

Examples:

```bash
acloudviewer list                    # show all assets in the current release
acloudviewer install --silent        # unattended desktop install
acloudviewer install --wheel         # pip-install the cloudViewer Python package
acloudviewer uninstall --yes         # remove the installation via maintenancetool
```

## Download behavior

Large assets (>= 64 MB) are fetched as **4 parallel ranged segments** written
into one preallocated file — release assets are served by a CDN that
throttles **per connection**, so segments download in parallel and aggregate
throughput roughly 2–3x (measured on the release CDN; override with
`--parallel 1..16`). Any stream that receives no bytes for 60s is aborted and
retried, and every retry **resumes from the bytes already on disk** (HTTP
`Range`) instead of restarting the whole file. If the server does not honor
range requests (a `200` reply to a ranged request), the downloader
automatically falls back to a single stream. Downloads land in a temp dir and
are removed after install.

## Uninstall

`uninstall` runs the QtIFW **maintenancetool** of the installation directory
(`purge --confirm-command`) — the same uninstaller the install wizard
provides, verified end-to-end. Without `--yes` it asks for confirmation on a
terminal; in a non-interactive shell (scripts, agents, CI) it **refuses** to
run unless `--yes` is given, so automation can never be surprised by it.

```bash
acloudviewer uninstall                 # interactive confirmation
acloudviewer uninstall --yes           # unattended
acloudviewer uninstall --prefix ~/opt/ACloudViewer --yes   # non-default location
```

## What it does per platform

| Platform | Default | `--silent` |
|----------|---------|-----------|
| Windows | Runs the QtIFW `.exe` wizard | QtIFW headless install to `--prefix` |
| macOS | Opens the `.dmg` in Finder (drag to Applications) | Runs the bundled installer headless (`hdiutil` attach → install → detach) |
| Linux | Runs the QtIFW `.run` wizard | QtIFW headless install to `--prefix` |

Silent installs mirror the pip harness's production-verified QtIFW console
invocations (`--root <dir> --accept-licenses --confirm-command install`), so
the same uninstaller/maintenance tooling applies. The GUI default keeps the
exact flow of downloading from the releases page.

## Troubleshooting: `acloudviewer: command not found`

`npm install -g` places the binary in npm's global `bin` directory
(`npm prefix -g`/`bin`). If that directory is not on your `PATH` (common with
custom npm prefixes), the command exists but the shell cannot find it:

```bash
export PATH="$(npm prefix -g)/bin:$PATH"   # add to ~/.bashrc to persist
```

## License

GPL-2.0-only — same as [ACloudViewer](https://github.com/Asher-1/ACloudViewer).
