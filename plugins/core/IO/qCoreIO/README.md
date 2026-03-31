# qCoreIO — Core engineering and binary I/O

## Introduction

**qCoreIO** provides import/export filters for ACloudViewer’s **native binary** format and several **engineering, hydraulics, and profile** text/binary formats. It is separate from generic PLY/XYZ handlers that may live in other core filters—this plugin’s filters are defined in `qCoreIO.cpp`.

## Supported formats

| Type | Formats |
|------|---------|
| Native | **BIN** — simple native binary cloud/container format (`SimpleBinFilter`). |
| Engineering / plant | **PDMS** — PDMS-like geometry (`PDMSFilter`). |
| Hydraulics | **MA**, **Mascaret** — hydraulics-related exchanges (`MAFilter`, `MascaretFilter`). |
| Profiles | **Height profiles** — height-profile text/binary (`HeightProfileFilter`). |

## Usage

Open and save these types from **File** menus when the plugin is installed. Use **`-FORMAT`** / **`-PRECISION`** on the CLI only with format identifiers your build’s writers accept.

## ACloudViewer CLI

```bash
ACloudViewer -SILENT -CORE_IO [-FORMAT <fmt>] [-PRECISION <n>] ...
```

| Flag | Description |
|------|-------------|
| `-CORE_IO` | Apply Core IO command-line options for subsequent operations. |
| `-FORMAT <fmt>` | Target format identifier for export (must match a supported core filter). |
| `-PRECISION <n>` | Numeric precision where the selected writer supports it. |

## Build

```bash
-DPLUGIN_IO_QCORE=ON
```

## Dependencies

No third-party SDK beyond ACloudViewer core/Qt for these filters.

## References

— (format documentation is domain-specific—PDMS, Mascaret, etc.)
