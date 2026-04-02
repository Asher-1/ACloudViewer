# Troubleshooting

## Windows CLI Hangs

The Windows `.bat` launchers use `start /b` so the GUI can detach in normal (graphical) mode. In **`-SILENT`** (headless) mode the same wrappers typically run the real executable in the **foreground** so logs and exit codes behave like a normal CLI.

If integration or pytest runs appear to **hang** when invoking the viewer, check **`ACV_BINARY`**: it should point to the **`.exe`** (or the real binary), not a **`.bat`** wrapper. Using the wrapper can leave processes or job control in a state that blocks the test runner.

## macOS Qt Platform Plugin

Do **not** set **`QT_QPA_PLATFORM`** when running the **`.app` bundle** on macOS. The application ships with the **cocoa** plugin only; forcing another platform (for example `xcb` or `offscreen` copied from Linux) prevents the GUI from starting or causes confusing plugin errors.

## Linux Missing Libraries

Headless Qt (`-SILENT`) on Linux often needs extra XCB libraries even without a display. Install at least:

- `libxcb-icccm4`
- `libxcb-shape0`

Example (Debian/Ubuntu):

```bash
sudo apt-get install libxcb-icccm4 libxcb-shape0
```

## Empty Output Files

Some **format conversions** can **fail silently** with **exit code 0**—for example exporting **STL** from data that cannot produce a valid mesh (e.g. point clouds without a proper mesh path). Always **assert on output file existence and non-zero size** in automation, not only on `returncode == 0`.
