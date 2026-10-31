// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OpenPCDetModelCatalog.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <cmath>
#include <cstring>

#include "QtCompat.h"

#ifdef AICore_ENABLED
#include "aicore/openpcdet_capi.h"
#endif

namespace OpenPCDetHelpers {

namespace {
// KITTI 3-class convention (OpenPCDet KITTI checkpoints): labels are
// 1-indexed. Other layouts fall back to a generic name because the AICore
// OpenPCDet C API exposes the class count only, never a name table.
constexpr const char* kKittiClassNames[3] = {"Car", "Pedestrian", "Cyclist"};

// Wireframe corner walk of the box corner layout in boxCorners():
// corner i keeps (lx, ly, lz) signs in bits 0/1/2 of i.
//   bottom cycle 0-1-3-2-0, verticals 0-4 / 1-5 / 3-7 / 2-6,
//   top cycle 4-5-7-6-4. 16 points / 15 segments cover all 12 edges
//   (4 edges drawn twice — a cube graph has no Eulerian path).
constexpr int kBoxWireIndices[16] = {0, 1, 3, 2, 0, 4, 5, 7,
                                     6, 4, 6, 2, 3, 7, 5, 1};
}  // namespace

QVector<OpenPCDetModelEntry> catalogModels() {
    QVector<OpenPCDetModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_openpcdet_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_openpcdet_model_entry* e = aicore_openpcdet_model_at(i);
        if (!e || !e->filename) continue;
        OpenPCDetModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.familyNote = QString::fromUtf8(e->family_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        out.append(entry);
    }
#else
    (void)0;
#endif
    return out;
}

int catalogDefaultIndex() {
#ifdef AICore_ENABLED
    return aicore_openpcdet_model_default_index();
#else
    return -1;
#endif
}

QStringList quantSuffixes() {
    return {QStringLiteral("f32"), QStringLiteral("f16"),
            QStringLiteral("q8_0")};
}

bool buildModelFilename(const QString& checkpoint,
                        const QString& quant,
                        QString* out) {
    if (checkpoint.isEmpty() || !quantSuffixes().contains(quant)) {
        return false;
    }
    if (out)
        *out = checkpoint + QStringLiteral(".") + quant +
               QStringLiteral(".gguf");
    return true;
}

InputKind inputKindForFilename(const QString& filename) {
    // The multi-modal checkpoints of the published release: CaDDN needs one
    // image + calibration, BEVFusion a 6-camera scene, and the focal VoxelRCNN
    // (multimodal) one image + calibration (its conv_focal_multimodal fusion
    // conv consumes concatenated image+voxel features; the AICore engine
    // rejects a points-only detect call for it, models/voxelrcnn.cpp).
    return filename.startsWith(QStringLiteral("kitti_caddn.")) ||
                           filename.startsWith(
                                   QStringLiteral("nuscenes_bevfusion.")) ||
                           filename.startsWith(QStringLiteral(
                                   "kitti_voxelrcnn_focal_multimodal."))
                   ? InputKind::Scene
                   : InputKind::Points;
}

QVector<OpenPCDetCheckpoint> catalogCheckpoints() {
    QVector<OpenPCDetCheckpoint> out;
    static const QRegularExpression kQuantTail(
            QStringLiteral("\\.(f32|f16|q8_0)\\.gguf$"));
    // The catalog appends " — <quant note>" (em dash, U+2014) to every
    // display name; matched via QChar so the split survives any source
    // encoding.
    static const QString kDisplaySep =
            QStringLiteral(" ") + QChar(0x2014) + QStringLiteral(" ");
    for (const OpenPCDetModelEntry& e : catalogModels()) {
        QString stem = e.filename;
        stem.remove(kQuantTail);
        if (stem.isEmpty() || stem == e.filename) continue;
        if (out.isEmpty() || out.last().stem != stem) {
            OpenPCDetCheckpoint cp;
            cp.stem = stem;
            // Strip the " — <quant note>" tail so the checkpoint label
            // stays quant-agnostic.
            const int sep = e.displayName.indexOf(kDisplaySep);
            cp.displayName = sep > 0 ? e.displayName.left(sep) : e.displayName;
            cp.familyNote = e.familyNote;
            out.append(cp);
        }
        out.last().quants.append(e);
    }
    return out;
}

int checkpointIndexForFilename(const QString& filename) {
    const QVector<OpenPCDetCheckpoint> checkpoints = catalogCheckpoints();
    for (int i = 0; i < checkpoints.size(); ++i) {
        for (const OpenPCDetModelEntry& e : checkpoints[i].quants) {
            if (e.filename == filename) return i;
        }
    }
    return -1;
}

bool findModelByFilename(const QString& filename, OpenPCDetModelEntry* out) {
    const QVector<OpenPCDetModelEntry> all = catalogModels();
    for (const OpenPCDetModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_openpcdet_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_openpcdet_free_buffer(dir);
        return out;
    }
#else
    (void)0;
#endif
    return QString();
}

QString modelDisplayLabel(const OpenPCDetModelEntry& entry) {
    QString label = entry.displayName;
    if (!entry.quantNote.isEmpty() && !label.contains(entry.quantNote)) {
        label += QStringLiteral(" ") + QChar(0x2014) + QStringLiteral(" ") +
                 entry.quantNote;
    }
    return label;
}

QString className(int32_t label, int numClass) {
    if (numClass == 3 && label >= 1 && label <= 3) {
        return QString::fromLatin1(kKittiClassNames[label - 1]);
    }
    return QStringLiteral("class %1").arg(label);
}

OpenPCDetClassColor classColor(int32_t label, int numClass) {
    // nuScenes devkit palette (1-indexed label = row index + 1 in the
    // OpenPCDet label convention: 1=car .. 10=traffic_cone).
    static constexpr unsigned char kNuScenes[10][3] = {
            {255, 158, 0},    // car
            {255, 99, 71},    // truck
            {233, 150, 70},   // construction_vehicle
            {255, 69, 0},     // bus
            {255, 140, 0},    // trailer
            {112, 128, 144},  // barrier
            {255, 61, 99},    // motorcycle
            {220, 20, 60},    // bicycle
            {0, 0, 230},      // pedestrian
            {0, 0, 110},      // traffic_cone
    };
    // Common KITTI demo colors.
    static constexpr unsigned char kKitti[3][3] = {
            {230, 60, 60},   // Car
            {60, 200, 60},   // Pedestrian
            {60, 120, 255},  // Cyclist
    };
    // Deterministic fallback cycle.
    static constexpr unsigned char kCycle[6][3] = {
            {255, 204, 0},   {80, 200, 255},  {170, 255, 80},
            {255, 105, 180}, {200, 160, 255}, {255, 130, 60},
    };
    OpenPCDetClassColor c;
    if (numClass == 10 && label >= 1 && label <= 10) {
        c.r = kNuScenes[label - 1][0];
        c.g = kNuScenes[label - 1][1];
        c.b = kNuScenes[label - 1][2];
    } else if (numClass == 3 && label >= 1 && label <= 3) {
        c.r = kKitti[label - 1][0];
        c.g = kKitti[label - 1][1];
        c.b = kKitti[label - 1][2];
    } else if (label >= 1) {
        const int idx = (label - 1) % 6;
        c.r = kCycle[idx][0];
        c.g = kCycle[idx][1];
        c.b = kCycle[idx][2];
    }
    return c;
}

bool pointInBox(const OpenPCDetDetection& det, float x, float y, float z) {
    // Translate to the box center and invert the heading rotation (KITTI
    // LiDAR frame, yaw about +Z) so membership becomes a half-size test.
    const double px = static_cast<double>(x) - det.x;
    const double py = static_cast<double>(y) - det.y;
    const double pz = static_cast<double>(z) - det.z;
    const double c = std::cos(-static_cast<double>(det.heading));
    const double s = std::sin(-static_cast<double>(det.heading));
    const double lx = px * c - py * s;
    const double ly = px * s + py * c;
    const double lz = pz;
    return std::abs(lx) <= static_cast<double>(det.dx) * 0.5 &&
           std::abs(ly) <= static_cast<double>(det.dy) * 0.5 &&
           std::abs(lz) <= static_cast<double>(det.dz) * 0.5;
}

std::map<int32_t, int> perClassCounts(
        const QVector<OpenPCDetDetection>& detections) {
    std::map<int32_t, int> counts;
    for (const OpenPCDetDetection& d : detections) {
        ++counts[d.label];
    }
    return counts;
}

QString csvHeader() {
    return QStringLiteral("x,y,z,dx,dy,dz,heading,score,label");
}

QString detectionToCsvRow(const OpenPCDetDetection& det) {
    return QStringLiteral("%1,%2,%3,%4,%5,%6,%7,%8,%9")
            .arg(det.x, 0, 'f', 4)
            .arg(det.y, 0, 'f', 4)
            .arg(det.z, 0, 'f', 4)
            .arg(det.dx, 0, 'f', 4)
            .arg(det.dy, 0, 'f', 4)
            .arg(det.dz, 0, 'f', 4)
            .arg(det.heading, 0, 'f', 4)
            .arg(det.score, 0, 'f', 4)
            .arg(det.label);
}

bool parsePointsBin(const QByteArray& bytes,
                    int pointDim,
                    std::vector<float>* outPoints,
                    int* outCount,
                    QString* error) {
    auto fail = [error](const QString& msg) {
        if (error) *error = msg;
        return false;
    };
    if (outPoints == nullptr || outCount == nullptr) {
        return fail(QStringLiteral("Internal error: null output"));
    }
    if (pointDim <= 0) {
        return fail(QStringLiteral("POINT_DIM must be positive, got %1")
                            .arg(pointDim));
    }
    const int floatBytes = static_cast<int>(sizeof(float)) * pointDim;
    if (bytes.isEmpty()) {
        return fail(QStringLiteral("Empty .bin file"));
    }
    if (bytes.size() % floatBytes != 0) {
        return fail(QStringLiteral(".bin size (%1 B) is not a multiple of %2 "
                                   "float32 features (%3 B)")
                            .arg(bytes.size())
                            .arg(pointDim)
                            .arg(floatBytes));
    }
    const int n = bytes.size() / floatBytes;
    outPoints->resize(static_cast<size_t>(n) * static_cast<size_t>(pointDim));
    // Single block copy; std::memcpy keeps it aliasing/alignment safe on
    // MSVC (no reinterpret of an unaligned buffer).
    std::memcpy(outPoints->data(), bytes.constData(),
                static_cast<size_t>(n) * static_cast<size_t>(floatBytes));
    *outCount = n;
    return true;
}

bool parsePointsTxt(const QString& text,
                    int pointDim,
                    std::vector<float>* outPoints,
                    int* outCount,
                    QString* error) {
    auto fail = [error](const QString& msg) {
        if (error) *error = msg;
        return false;
    };
    if (outPoints == nullptr || outCount == nullptr) {
        return fail(QStringLiteral("Internal error: null output"));
    }
    if (pointDim <= 0) {
        return fail(QStringLiteral("POINT_DIM must be positive, got %1")
                            .arg(pointDim));
    }
    outPoints->clear();
    *outCount = 0;

    static const QRegularExpression kWhitespace(QStringLiteral("\\s+"));
    const QStringList lines = text.split(QChar('\n'));
    outPoints->reserve(lines.size() * static_cast<size_t>(pointDim));
    int lineNo = 0;
    for (const QString& rawLine : lines) {
        ++lineNo;
        const QString line = rawLine.trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) continue;
        const QStringList tokens =
                line.split(kWhitespace, QtCompat::SkipEmptyParts);
        if (tokens.size() < 3) {
            return fail(
                    QStringLiteral("Line %1: expected 'x y z [i]', got '%2'")
                            .arg(lineNo)
                            .arg(line));
        }
        for (int f = 0; f < pointDim; ++f) {
            if (f < tokens.size()) {
                bool ok = false;
                const float v = tokens.at(f).toFloat(&ok);
                if (!ok) {
                    return fail(QStringLiteral("Line %1: '%2' is not a number")
                                        .arg(lineNo)
                                        .arg(tokens.at(f)));
                }
                outPoints->push_back(v);
            } else {
                outPoints->push_back(0.0f);  // pad missing features
            }
        }
    }
    if (outPoints->empty()) {
        return fail(QStringLiteral("No points found in the .txt file"));
    }
    *outCount = static_cast<int>(outPoints->size() / pointDim);
    return true;
}

void boxCorners(const OpenPCDetDetection& det, float* outCorners24) {
    const float hx = 0.5f * det.dx;
    const float hy = 0.5f * det.dy;
    const float hz = 0.5f * det.dz;
    const float c = std::cos(det.heading);
    const float s = std::sin(det.heading);
    for (int i = 0; i < 8; ++i) {
        const float lx = (i & 1) ? hx : -hx;
        const float ly = (i & 2) ? hy : -hy;
        const float lz = (i & 4) ? hz : -hz;
        // KITTI LiDAR frame: yaw (heading) rotates about +Z.
        outCorners24[3 * i + 0] = det.x + lx * c - ly * s;
        outCorners24[3 * i + 1] = det.y + lx * s + ly * c;
        outCorners24[3 * i + 2] = det.z + lz;
    }
}

std::vector<EdgePair> boxEdges() {
    // Bottom 0-1-3-2, top 4-5-7-6, verticals 0-4 / 1-5 / 3-7 / 2-6.
    return {{0, 1}, {1, 3}, {3, 2}, {2, 0}, {4, 5}, {5, 7},
            {7, 6}, {6, 4}, {0, 4}, {1, 5}, {3, 7}, {2, 6}};
}

std::vector<int> boxWireIndices() {
    return std::vector<int>(std::begin(kBoxWireIndices),
                            std::end(kBoxWireIndices));
}

QImage sceneCameraPreview(const float* images,
                          int numCameras,
                          int height,
                          int width,
                          int channels,
                          int camera) {
    if (images == nullptr || numCameras <= 0 || height <= 0 || width <= 0 ||
        camera < 0 || camera >= numCameras) {
        return QImage();
    }
    const int usedChannels =
            (channels == 1 || channels == 3 || channels == 4) ? channels : 0;
    if (usedChannels == 0) return QImage();
    QImage image(width, height, QImage::Format_RGB888);
    const float* frame =
            images + static_cast<size_t>(camera) * height * width * channels;
    for (int y = 0; y < height; ++y) {
        uchar* row = image.scanLine(y);
        for (int x = 0; x < width; ++x) {
            const float* px =
                    frame + (static_cast<size_t>(y) * width + x) * channels;
            const auto clamp8 = [](float v) {
                return static_cast<uchar>(qBound(0.0f, v * 255.0f, 255.0f) +
                                          0.5f);
            };
            if (usedChannels == 1) {
                const uchar g = clamp8(px[0]);
                row[3 * x + 0] = g;
                row[3 * x + 1] = g;
                row[3 * x + 2] = g;
            } else {
                row[3 * x + 0] = clamp8(px[0]);
                row[3 * x + 1] = clamp8(px[1]);
                row[3 * x + 2] = clamp8(px[2]);
            }
        }
    }
    return image;
}

bool readScenePreview(const QString& sceneDir, QImage* out, QString* error) {
    auto fail = [error](const QString& msg) {
        if (error) *error = msg;
        return false;
    };
    const QFileInfo info(sceneDir);
    if (!info.isDir()) {
        return fail(QStringLiteral("Not a scene directory: %1").arg(sceneDir));
    }
    // scene.json: num_cameras / image_height / image_width / image_channels.
    // Plain key lookup (same semantics as the AICore scene loader), no
    // regex: escaped-pattern literals trip the boundary checker's
    // absolute-path heuristic.
    QFile json(QDir(sceneDir).filePath(QStringLiteral("scene.json")));
    if (!json.open(QIODevice::ReadOnly)) {
        return fail(QStringLiteral("Missing scene.json in %1").arg(sceneDir));
    }
    const QByteArray payload = json.readAll();
    int numCameras = 0, height = 0, width = 0, channels = 0;
    const auto grab = [&payload](const char* key, int* value) {
        const QByteArray needle =
                QByteArrayLiteral("\"") + key + QByteArrayLiteral("\"");
        const int pos = payload.indexOf(needle);
        if (pos < 0) return;
        const int colon = payload.indexOf(':', pos);
        if (colon < 0) return;
        // Skip whitespace, then take the decimal digit run (the value may
        // be followed by a comma / next key).
        int start = colon + 1;
        while (start < payload.size() &&
               (payload[start] == ' ' || payload[start] == '\t')) {
            ++start;
        }
        int end = start;
        while (end < payload.size() && payload[end] >= '0' &&
               payload[end] <= '9') {
            ++end;
        }
        if (end > start) *value = payload.mid(start, end - start).toInt();
    };
    grab("num_cameras", &numCameras);
    grab("image_height", &height);
    grab("image_width", &width);
    grab("image_channels", &channels);
    if (numCameras <= 0 || height <= 0 || width <= 0 || channels <= 0) {
        return fail(
                QStringLiteral("Incomplete scene.json in %1").arg(sceneDir));
    }
    QFile images(QDir(sceneDir).filePath(QStringLiteral("images.bin")));
    if (!images.open(QIODevice::ReadOnly)) {
        return fail(QStringLiteral("Missing images.bin in %1").arg(sceneDir));
    }
    const qint64 frameBytes =
            static_cast<qint64>(height) * width * channels * sizeof(float);
    const QByteArray frame = images.read(frameBytes);
    if (frame.size() < frameBytes) {
        return fail(
                QStringLiteral("images.bin is truncated in %1").arg(sceneDir));
    }
    QImage preview = sceneCameraPreview(
            reinterpret_cast<const float*>(frame.constData()), numCameras,
            height, width, channels, 0);
    if (preview.isNull()) {
        return fail(QStringLiteral("Unsupported scene payload in %1")
                            .arg(sceneDir));
    }
    if (out) *out = preview;
    return true;
}

OpenPCDetTestDataFiles testDataFilesForCheckpoint(const QString& checkpoint) {
    OpenPCDetTestDataFiles out;
    if (checkpoint.isEmpty()) return out;
    const bool scene =
            inputKindForFilename(checkpoint + QStringLiteral(".f16.gguf")) ==
            InputKind::Scene;
    const bool nuscenes = checkpoint.startsWith(QStringLiteral("nuscenes"));
    if (scene) {
        if (nuscenes) {
            // BEVFusion: 6-camera scene + the nuScenes LiDAR branch.
            out.sceneDirName = QStringLiteral("scene_bevfusion");
            out.scenePointsFile = QStringLiteral("nuscenes_n015_lidar.bin");
        } else {
            // CaDDN and the focal VoxelRCNN (multimodal): the KITTI
            // single-camera scene + the raw frame as LiDAR branch (the
            // same pairing the AICore focal-scene scenario runs with).
            out.sceneDirName = QStringLiteral("scene_caddn");
            out.scenePointsFile = QStringLiteral("kitti_000008.bin");
        }
        return out;
    }
    if (nuscenes) {
        out.pointsFile = QStringLiteral("nuscenes_n015_lidar.bin");
    } else if (checkpoint.contains(QStringLiteral("pointrcnn"))) {
        // PointRCNN checkpoints train on the pre-sampled 16384-point
        // variant of the same frame.
        out.pointsFile = QStringLiteral("kitti_000008_sampled16384.bin");
    } else {
        out.pointsFile = QStringLiteral("kitti_000008.bin");
    }
    return out;
}

bool resolveTestDataPaths(const QString& bundleRoot,
                          const QString& checkpoint,
                          OpenPCDetTestDataFiles* out,
                          QString* error) {
    auto fail = [error](const QString& msg) {
        if (error) *error = msg;
        return false;
    };
    if (out) *out = OpenPCDetTestDataFiles{};
    if (bundleRoot.isEmpty() || !QDir(bundleRoot).exists()) {
        return fail(QStringLiteral("Test-data bundle not extracted: %1")
                            .arg(bundleRoot));
    }
    const OpenPCDetTestDataFiles files = testDataFilesForCheckpoint(checkpoint);
    if (files.pointsFile.isEmpty() && files.sceneDirName.isEmpty()) {
        return fail(QStringLiteral("No test-data mapping for checkpoint %1")
                            .arg(checkpoint));
    }
    const QDir root(bundleRoot);
    const auto require = [&fail, &root](const QString& rel) -> QString {
        const QString path = root.filePath(rel);
        if (!QFileInfo::exists(path)) {
            fail(QStringLiteral("Missing test-data member: %1").arg(rel));
            return QString();
        }
        return QFileInfo(path).absoluteFilePath();
    };
    OpenPCDetTestDataFiles resolved;
    if (!files.pointsFile.isEmpty()) {
        resolved.pointsFile = require(files.pointsFile);
        if (resolved.pointsFile.isEmpty()) return false;
    }
    if (!files.sceneDirName.isEmpty()) {
        resolved.sceneDirName = require(files.sceneDirName);
        if (resolved.sceneDirName.isEmpty()) return false;
    }
    if (!files.scenePointsFile.isEmpty()) {
        resolved.scenePointsFile = require(files.scenePointsFile);
        if (resolved.scenePointsFile.isEmpty()) return false;
    }
    if (out) *out = resolved;
    return true;
}

}  // namespace OpenPCDetHelpers
