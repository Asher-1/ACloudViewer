# qCSVMatrixIO — CSV matrix grids

## Introduction

**qCSVMatrixIO** imports and exports **scalar or height matrices** as **CSV** text: each cell maps to a grid sample (row/column layout), useful for DEM-like surfaces and heatmap-style data.

## Supported formats

| Format | Notes |
|--------|--------|
| **CSV matrix** | Delimited text grid; optional header row; optional row inversion. |

## Usage

Open or save matrix CSV from the GUI when the filter is registered. Match **separator**, **header**, and **row order** to your data convention.

## ACloudViewer CLI

```bash
ACloudViewer -SILENT -CSV_MATRIX [-SEPARATOR <char>] [-SKIP_HEADER] [-INVERT_ROWS] ...
```

| Flag | Description |
|------|-------------|
| `-CSV_MATRIX` | Use the CSV matrix I/O command path. |
| `-SEPARATOR <char>` | Field delimiter (e.g. `,`, `;`, tab). |
| `-SKIP_HEADER` | Treat the first line as a non-data header. |
| `-INVERT_ROWS` | Invert row order when reading/writing. |

## Build

```bash
-DPLUGIN_IO_QCSV_MATRIX=ON
```

## Dependencies

None beyond ACloudViewer core/Qt as used by the plugin (text parsing only).

## References

— (no external specification; format is project-defined per CSV layout.)
