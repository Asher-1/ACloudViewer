// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QMetaType>
#include <QString>
#include <QStringList>
#include <QVector>
#include <cstdint>
#include <map>
#include <vector>

/** One detected 3D box (LiDAR frame), mirroring
 *  aicore_openpcdet_detection. */
struct OpenPCDetDetection {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float heading = 0.0f;
    float score = 0.0f;
    /** 1-indexed class (OpenPCDet convention). */
    int32_t label = 0;
    /** Human-readable class name (OpenPCDetHelpers::className). */
    QString className;
};

Q_DECLARE_METATYPE(OpenPCDetDetection)

/** Result envelope of one OpenPCDet inference. */
struct OpenPCDetRunResult {
    QString sourceLabel;    ///< cloud name / input file stem / scene dir name
    QString modelPath;
    QString resolvedDevice; ///< backend-resolved device (may differ from request)
    double runtimeMs = 0.0;
    int numPoints = 0;
    int pointDim = 0;       ///< point feature count handed to the engine
    int numClass = 0;       ///< class count declared by the model geometry
    QVector<OpenPCDetDetection> detections;
    uint32_t timingValidFields = 0;
    double inferenceMs = 0.0;
    double preprocessMs = 0.0;
    double postprocessMs = 0.0;
    double e2eMs = 0.0;
    // ---- multi-modal scene runs (detect_scene) ----
    bool fromScene = false; ///< true when the run went through detect_scene
    QString sceneDir;       ///< scene directory (empty on point-only runs)
    int numCameras = 0;     ///< camera count of the loaded scene
};

Q_DECLARE_METATYPE(OpenPCDetRunResult)

/** Geometry summary mirroring aicore_openpcdet_geometry. */
struct OpenPCDetGeometry {
    int numPointFeatures = 0;
    int gridSizeX = 0;
    int gridSizeY = 0;
    int gridSizeZ = 0;
    int numClass = 0;
    float voxelSizeX = 0.0f;
    float voxelSizeY = 0.0f;
    float voxelSizeZ = 0.0f;
    float pointCloudRange[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    QString arch;
};

/** Catalog entry mirroring aicore_openpcdet_model_entry. */
struct OpenPCDetModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString familyNote;
    QString licenseNote;
};

namespace OpenPCDetHelpers {

/** Enumerate the published catalog from AICore. */
QVector<OpenPCDetModelEntry> catalogModels();

/** Input routing of a catalog GGUF: the multi-modal checkpoints
 *  (kitti_caddn.* / nuscenes_bevfusion.* / kitti_voxelrcnn_focal_multimodal.*)
 *  run through detect_scene with a scene directory; every other checkpoint
 *  runs detect_points on a point buffer. Pure mapping — unit tested without
 *  AICore. */
enum class InputKind { Points, Scene };
InputKind inputKindForFilename(const QString& filename);

/** The three published quantization suffixes, in catalog order
 *  (f32 / f16 / q8_0). */
QStringList quantSuffixes();

/** Catalog GGUF filename of a checkpoint stem + quantization suffix
 *  ("kitti_pointpillar" + "f16" -> "kitti_pointpillar.f16.gguf"). Returns
 *  false on an empty stem or a quant not listed by quantSuffixes(). Pure
 *  mapping — unit tested without AICore. */
bool buildModelFilename(const QString& checkpoint, const QString& quant,
                        QString* out);

/** One detector checkpoint (18 published) with its three quantized
 *  catalog rows. */
struct OpenPCDetCheckpoint {
    QString stem;        ///< "kitti_pointpillar", ...
    QString displayName; ///< display name of any quantized row, quant note stripped
    QString familyNote;
    QVector<OpenPCDetModelEntry> quants;  ///< f32 / f16 / q8_0 rows
};

/** Group the 54 catalog rows into the 18 detector checkpoints, ordered as
 *  the AICore catalog publishes them. Empty when AICore is not available. */
QVector<OpenPCDetCheckpoint> catalogCheckpoints();

/** Index (within catalogCheckpoints()) of the checkpoint that owns
 *  \p filename; -1 when unknown. */
int checkpointIndexForFilename(const QString& filename);

/** Index of the catalog-declared default row
 *  (aicore_openpcdet_model_default_index); -1 when AICore is not
 *  available. */
int catalogDefaultIndex();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, OpenPCDetModelEntry* out);

/** Model cache directory for qOpenPCDet
 *  (aicore_openpcdet_model_cache_dir). */
QString modelCacheDir();

/** Build the user-facing catalog label without duplicating a quantization
 *  note that is already part of displayName. */
QString modelDisplayLabel(const OpenPCDetModelEntry& entry);

/** Human-readable class name for a 1-indexed label. The AICore OpenPCDet
 *  C API exposes the label index only (no class-name table), so the
 *  KITTI 3-class convention (1=Car, 2=Pedestrian, 3=Cyclist) is applied
 *  when the model declares exactly 3 classes; any other layout falls
 *  back to "class N". Pure mapping — unit tested without AICore. */
QString className(int32_t label, int numClass);

/** RGB tuple of the per-class visualization palette (0-255 per channel),
 *  mirroring the colors used by the OpenPCDet Python demos: the nuScenes
 *  devkit palette when the model declares 10 classes and the common
 *  KITTI demo colors (Car=red, Pedestrian=green, Cyclist=blue) at 3
 *  classes; other layouts fall back to a deterministic 6-color cycle.
 *  Pure mapping — unit tested without AICore. */
struct OpenPCDetClassColor {
    unsigned char r = 255;
    unsigned char g = 204;
    unsigned char b = 0;
};
OpenPCDetClassColor classColor(int32_t label, int numClass);

/** True when the point (x,y,z) lies inside the oriented detection box
 *  (heading rotation about +Z inverted around the box center, then a
 *  half-size membership test per axis). Pure geometry — unit tested
 *  without AICore. */
bool pointInBox(const OpenPCDetDetection& det, float x, float y, float z);

/** Per-label detection counts keyed by the 1-indexed label. */
std::map<int32_t, int> perClassCounts(
        const QVector<OpenPCDetDetection>& detections);

/** CSV column header of the detection export (x,y,z,dx,dy,dz,heading,
 *  score,label). */
QString csvHeader();
/** One CSV row for a detection (no trailing newline). */
QString detectionToCsvRow(const OpenPCDetDetection& det);

/** Parses raw KITTI-style .bin bytes as row-major float32
 *  [n_points, pointDim]. The byte count must be a non-zero multiple of
 *  pointDim * 4. Returns false with \p error set otherwise. Pure parsing
 *  — unit tested without AICore. */
bool parsePointsBin(const QByteArray& bytes,
                    int pointDim,
                    std::vector<float>* outPoints,
                    int* outCount,
                    QString* error);

/** Parses whitespace-separated ASCII points ("x y z [i]" per line;
 *  '#' comment lines and blank lines are skipped). Missing trailing
 *  features are zero-padded to \p pointDim. Returns false with \p error
 *  set on a malformed line. Pure parsing — unit tested without AICore. */
bool parsePointsTxt(const QString& text,
                    int pointDim,
                    std::vector<float>* outPoints,
                    int* outCount,
                    QString* error);

/** Box corner layout used by the wireframe export: corner i keeps
 *  (lx, ly, lz) signs from bits 0/1/2 of i (bit set = +half size), then
 *  rotates the xy offsets by \p heading around the box center (KITTI
 *  LiDAR frame, yaw about +Z). Writes 8 corners (x,y,z interleaved) to
 *  \p outCorners24. Pure math — unit tested without AICore. */
void boxCorners(const OpenPCDetDetection& det, float* outCorners24);

/** The 12 cube edges as corner-index pairs for the corner layout of
 *  boxCorners() (4 bottom, 4 top, 4 verticals). */
struct EdgePair {
    int a;
    int b;
};
std::vector<EdgePair> boxEdges();

/** Corner-index walk drawing all 12 edges of the box as ONE open
 *  polyline (a cube graph has 8 odd-degree vertices, so a strict
 *  Eulerian path does not exist; 4 edges are intentionally traced
 *  twice — visually identical, one DB entity per box). Unit tested to
 *  cover exactly the 12 unique edges. */
std::vector<int> boxWireIndices();

/** Converts one camera frame of the NHWC float32 [num_cameras, h, w,
 *  channels] scene payload (aicore_openpcdet_load_scene_dir layout) into a
 *  preview QImage (RGB888; 1/3/4-channel inputs accepted, values assumed
 *  0..1 and clamped). UI preview only — never an inference path. Pure
 *  conversion — unit tested without AICore. */
QImage sceneCameraPreview(const float* images,
                          int numCameras,
                          int height,
                          int width,
                          int channels,
                          int camera);

/** Reads scene.json dimensions + the first camera frame of images.bin for
 *  the scene-directory preview. Returns false with \p error set when the
 *  directory does not carry a readable scene payload. */
bool readScenePreview(const QString& sceneDir, QImage* out, QString* error);

/** Test-data file mapping of one checkpoint for the "Use test data" fill
 *  (the shared openpcdet_test_data bundle). Bundle-relative names; empty
 *  members are not filled by the dialog. Pure mapping — unit tested
 *  without AICore. */
struct OpenPCDetTestDataFiles {
    /** LiDAR .bin of the point-cloud panels (kitti_000008.bin, its
     *  PointRCNN pre-sampled 16384-point variant, or the nuScenes
     *  5-feature sweep). */
    QString pointsFile;
    /** Scene directory of the multi-modal panels (scene_caddn /
     *  scene_bevfusion). */
    QString sceneDirName;
    /** LiDAR-branch point file of the multi-modal panels (required by
     *  BEVFusion and the focal VoxelRCNN). */
    QString scenePointsFile;
};

/** Bundle-relative test-data files the panel of \p checkpoint consumes:
 *  KITTI point panels take the raw 000008 frame (PointRCNN checkpoints its
 *  pre-sampled 16384-point variant), nuScenes point panels the n015
 *  sweep, KITTI scene panels (CaDDN + focal multimodal) the CaDDN scene
 *  with the raw frame as LiDAR branch, BEVFusion the 6-camera scene with
 *  the nuScenes sweep. Pure mapping — unit tested without AICore. */
OpenPCDetTestDataFiles testDataFilesForCheckpoint(const QString& checkpoint);

/** Resolves the bundle-relative mapping of \p checkpoint into absolute
 *  paths under \p bundleRoot (the extracted openpcdet_test_data group
 *  directory). Returns false with \p error set when a needed member is
 *  missing — the dialog's error path. Unit tested without AICore. */
bool resolveTestDataPaths(const QString& bundleRoot,
                          const QString& checkpoint,
                          OpenPCDetTestDataFiles* out,
                          QString* error);

}  // namespace OpenPCDetHelpers
