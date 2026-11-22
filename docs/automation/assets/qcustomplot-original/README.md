# Original QCustomPlot API examples

These 29 PNGs are newly authored functionality illustrations for the QCustomPlot
1.x API bundled with ACloudViewer. Every picture is visibly labelled “Original
example”. They are **not recovered upstream release images**, and are not copies
or tracings of the official documentation artwork. The API documentation text
and historical release HTML remain unchanged.

The deterministic numeric data, annotations, palettes and layouts are defined in
`main.cpp`. The renderer compiles the repository's existing QCustomPlot sources
in an independent Qt target; it does not link an application build directory.
The generated diagrams illustrate the meaning of the corresponding API objects,
not the pixel layout of the original missing illustrations.

## Reproduce

Prerequisites: CMake >=3.16, C++17 compiler, Qt5 Core/Gui/Widgets/PrintSupport,
DejaVu Sans fonts and Python with Pillow. No network access is needed.

```bash
ACV_SOURCE_ROOT=/absolute/path/to/ACloudViewer \
  QT_PREFIX=/path/to/Qt5 bash build-and-render.sh
```

The script limits its independent build to two workers, writes PNGs into
`output/`, and losslessly recompresses them. `optimize_pngs.py` verifies decoded
RGBA pixels before/after compression. PNG bytes can vary with Qt/font/library
versions; the committed `manifest.json` records the reviewed publication bytes.
Changing or regenerating images requires updating and reviewing that manifest.

## Publication and history

`docs/automation/scripts/install_qcp_doc_images.py` is the single publisher for
both a newly generated API tree and historical release overlays. It reads actual
HTML image references, validates the corresponding source SHA256, and fills only
missing local image paths. It does not replace existing historical images, alter
HTML, fetch external files or fabricate an unknown illustration. It publishes
this README and the adjacent license notices beside repaired API pages.

The renderer and examples were checked against the repository's bundled API,
whose relevant image-bearing documentation blocks match upstream 1.3.1. The
historical v3.9.4 library header has the same API source bytes. New examples are
explicitly distinguished from original historical artwork.

See `LICENSE-NOTICE.md` and `LICENSE`: original rendered diagrams are contributed
under the project's GPL-2.0-or-later license; the generator's GPL-3.0-or-later notice and the
underlying QCustomPlot licensing obligations are separate. Nothing here
relicenses third-party code or the official documentation's artwork.
