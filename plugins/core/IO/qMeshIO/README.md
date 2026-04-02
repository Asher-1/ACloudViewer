# qMeshIO — Assimp-based mesh import

## Introduction

**qMeshIO** imports **3D mesh** data via the **Open Asset Import Library (Assimp)**. This tree enables a focused set of Assimp importers (see **Supported formats**); geometry is translated into ACloudViewer mesh entities. Problems in specific files may originate in Assimp or in the translation layer—report minimal reproducers upstream when isolated.

## Supported formats

The CMake configuration for this plugin turns on these Assimp importers:

| Format | Typical extensions |
|--------|---------------------|
| **COLLADA** | `.dae` |
| **glTF** | `.gltf`, `.glb` |
| **IFC (IFC-SPF)** | `.ifc` (often described with STEP-related exchange; importer is IFC-specific here) |

Other mesh types (e.g. generic OBJ/STL via Assimp) are **not** enabled in the current `CMakeLists.txt`; extend importer flags only if you add and test them.

## Usage

Open supported mesh files through **File → Import**. Mesh names or hierarchy may be simplified on import. IFC coverage is not complete for all vendor exports—see Assimp issue trackers for edge cases.

## ACloudViewer CLI

```bash
ACloudViewer -SILENT -MESH_IO [-SCALE <factor>] [-UP_AXIS X|Y|Z] [-MERGE_NODES] ...
```

| Flag | Description |
|------|-------------|
| `-MESH_IO` | Activate Mesh IO command-line options for the following arguments. |
| `-SCALE <factor>` | Positive scale factor applied to imported mesh coordinates. |
| `-UP_AXIS` | `X`, `Y`, or `Z` — up axis hint for loaders that use it. |
| `-MERGE_NODES` | Enable vertex merge where the pipeline supports it. |

## Build

```bash
-DPLUGIN_IO_QMESH=ON
```

Provide **Assimp** include and library directories (`ASSIMP_INCLUDE_DIR`, `ASSIMP_LIB_DIR`, `ASSIMP_LIBRARIES`) as required by your layout.

## Dependencies

- **[Assimp](https://github.com/assimp/assimp)** — Open Asset Import Library (importers selected in this plugin’s `CMakeLists.txt`).

## References

- Assimp: [https://github.com/assimp/assimp](https://github.com/assimp/assimp)
- Upstream MeshIO lineage: Andy Maloney’s MeshIO plugin for CloudCompare (BSD-3-Clause).
