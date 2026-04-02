# qStepCADImport — STEP CAD import

## Introduction

**qStepCADImport** imports **STEP** CAD models (**ISO 10303**) into ACloudViewer: B-Rep bodies tessellated for visualization and measurement next to point clouds. Units and axis conventions follow the STEP file and **Open CASCADE** import—apply transforms in the viewer if CRS does not match your survey.

## Supported formats

| Format | Notes |
|--------|--------|
| **STEP / STP** | Exchange per ISO 10303; applicable AP subsets depend on OCCT build (e.g. AP203, AP214, AP242 features). |

## Usage

Use **File → Import** for STEP. Large assemblies can be slow to tessellate; simplified bodies or pre-meshed exports may be faster. For unattended batch, prefer external CAD converters or automation that drives the GUI if your build exposes it.

## ACloudViewer CLI

**None** — STEP import is not exposed as a dedicated `-SILENT` command in this plugin.

## Build

```bash
-DPLUGIN_IO_QSTEP=ON \
  -DOPENCASCADE_INC_DIR=/path/to/occt/include \
  -DOPENCASCADE_LIB_DIR=/path/to/occt/lib
```

On Windows, also set **`OPENCASCADE_DLL_DIR`** for install/runtime deployment.

## Dependencies

- **Open CASCADE Technology (OCCT)** — STEP readers and meshing toolkits (`TKSTEP`, `TKSTEPBase`, … as listed in the plugin `CMakeLists.txt`).

## References

- OCCT: [https://dev.opencascade.org/](https://dev.opencascade.org/)
- ISO 10303 STEP overview.
