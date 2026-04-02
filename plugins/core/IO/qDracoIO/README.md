# qDracoIO — Google Draco compressed geometry

## Introduction

**qDracoIO** reads and writes **Google Draco** `.drc` files: compressed triangular meshes and point clouds suitable for transmission and storage. Quantization and encoder settings trade size against fidelity.

## Supported formats

| Format | Notes |
|--------|--------|
| **DRC** | Draco-compressed mesh or point cloud (Google Draco bitstream). |

## Usage

Open and save `.drc` through the normal file dialogs or silent batch flows. Tune **quantization**, **compression level**, and **speed** in the GUI or via CLI flags below to match accuracy and file-size targets.

## ACloudViewer CLI

```bash
ACloudViewer -SILENT -DRACO [-QUANTIZATION <n>] [-COMPRESSION_LEVEL <n>] [-SPEED <n>] ...
```

| Flag | Description |
|------|-------------|
| `-DRACO` | Use the Draco encode/decode path. |
| `-QUANTIZATION <n>` | Quantization bits for positions/attributes (library-dependent ranges). |
| `-COMPRESSION_LEVEL <n>` | Encoder compression level. |
| `-SPEED <n>` | Encoder speed preset (implementation-dependent). |

Numeric limits follow the **Draco** version linked into your build.

## Build

```bash
-DPLUGIN_IO_QDRACO=ON
```

Point CMake at **Draco** so `DRACO_INCLUDE_DIRS` / `DRACO_TARGET` (or equivalent) resolve.

## Dependencies

- **[Google Draco](https://google.github.io/draco/)** — Encoder and decoder libraries.

## References

- Draco: [https://google.github.io/draco/](https://google.github.io/draco/)
