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
#include <QFont>
#include <QFontMetrics>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPen>
#include <QRegularExpression>
#include <algorithm>
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

// nuScenes 10-class convention, 1-indexed, in the OpenPCDet CLASS_NAMES
// order the engine actually emits. The converter writes
// label_indices = CLASS_NAMES.index(name) + 1 and the PyTorch reference
// uses the same mapping, so label N is the N-th entry of the upstream
// nuscenes config CLASS_NAMES (car, truck, construction_vehicle, bus,
// trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone).
// NOTE: the upstream gen_bev_nuscenes.py LABEL_NAMES_NUSCENES display
// table is inconsistent with that label order; do NOT copy it.
constexpr const char* kNuScenesClassNames[10] = {
        "car",        "truck",       "construction_vehicle", "bus",
        "trailer",    "barrier",     "motorcycle",           "bicycle",
        "pedestrian", "traffic_cone"};

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
    if (numClass == 10 && label >= 1 && label <= 10) {
        return QString::fromLatin1(kNuScenesClassNames[label - 1]);
    }
    if (numClass == 1 && label == 1) {
        // The only published single-class checkpoint is the car-only
        // KITTI Voxel-RCNN (see the AICore openpcdet model catalog).
        return QStringLiteral("Car");
    }
    return QStringLiteral("class %1").arg(label);
}

OpenPCDetClassColor classColor(int32_t label, int numClass) {
    // Upstream OpenPCDet-GGML demo palette (ColorBrewer Set1, indexed by
    // the 1-based label): gen_bev_kitti.py LABEL_COLORS = {1: '#e41a1c',
    // 2: '#377eb8', 3: '#4daf4a'} and gen_bev_nuscenes.py extends the same
    // table to the 10 nuScenes classes. Label-driven, like the upstream
    // scripts; numClass only documents the model layout.
    static constexpr unsigned char kUpstreamSet1[10][3] = {
            {228, 26, 28},    // 1  car / Car                  #e41a1c
            {55, 126, 184},   // 2  truck / Pedestrian         #377eb8
            {77, 175, 74},    // 3  construction_vehicle / Cyc #4daf4a
            {152, 78, 163},   // 4  bus                        #984ea3
            {255, 127, 0},    // 5  trailer                    #ff7f00
            {166, 86, 40},    // 6  barrier                    #a65628
            {247, 129, 191},  // 7  motorcycle                 #f781bf
            {153, 153, 153},  // 8  bicycle                    #999999
            {102, 194, 165},  // 9  pedestrian                 #66c2a5
            {252, 141, 98},   // 10 traffic_cone               #fc8d62
    };
    // Deterministic fallback for labels outside the published range.
    static constexpr unsigned char kCycle[6][3] = {
            {255, 204, 0},   {80, 200, 255},  {170, 255, 80},
            {255, 105, 180}, {200, 160, 255}, {255, 130, 60},
    };
    OpenPCDetClassColor c;
    if (label >= 1 && label <= 10) {
        c.r = kUpstreamSet1[label - 1][0];
        c.g = kUpstreamSet1[label - 1][1];
        c.b = kUpstreamSet1[label - 1][2];
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

namespace {

// Inverse of the upper-left 3x3 of a row-major 4x4 matrix, computed in
// double precision (the engine's mat3_inverse chain). Returns false on a
// singular rotation.
bool mat3Inverse(const float* m44, double* inv9) {
    const double a = m44[0], b = m44[1], c = m44[2];
    const double d = m44[4], e = m44[5], f = m44[6];
    const double g = m44[8], h = m44[9], i = m44[10];
    const double det =
            a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::abs(det) < 1e-12) return false;
    const double s = 1.0 / det;
    inv9[0] = (e * i - f * h) * s;
    inv9[1] = (c * h - b * i) * s;
    inv9[2] = (b * f - c * e) * s;
    inv9[3] = (f * g - d * i) * s;
    inv9[4] = (a * i - c * g) * s;
    inv9[5] = (c * d - a * f) * s;
    inv9[6] = (d * h - e * g) * s;
    inv9[7] = (b * g - a * h) * s;
    inv9[8] = (a * e - b * d) * s;
    return true;
}

// Liang-Barsky segment clip against the pixel rectangle [0, w-1] x
// [0, h-1] (adapted from qAnyLabeling PointCloudCamera.cpp clipSegment —
// single consumer here, so the math stays plugin-local).
bool clipSegmentToRect(const QPointF& start,
                       const QPointF& end,
                       int width,
                       int height,
                       QLineF* result) {
    const double dx = end.x() - start.x();
    const double dy = end.y() - start.y();
    double lower = 0.0;
    double upper = 1.0;
    const double p[4] = {-dx, dx, -dy, dy};
    const double q[4] = {start.x(), width - 1.0 - start.x(), start.y(),
                         height - 1.0 - start.y()};
    for (int k = 0; k < 4; ++k) {
        if (p[k] == 0.0) {
            if (q[k] < 0.0) return false;
            continue;
        }
        const double ratio = q[k] / p[k];
        if (p[k] < 0.0)
            lower = std::max(lower, ratio);
        else
            upper = std::min(upper, ratio);
        if (lower > upper) return false;
    }
    *result = QLineF(start + (end - start) * lower,
                     start + (end - start) * upper);
    return true;
}

// Positional camera-name fallback of legacy scenes without the optional
// scene.json "camera_names" array: the published bundles are KITTI
// single-camera (CAM2 / image_02) and the nuScenes 6-camera ring in the
// upstream infer_bevfusion_reference.py CAMERAS order.
QString defaultCameraName(int index, int count) {
    if (count == 1) return QStringLiteral("CAM2");
    if (count == 6) {
        static const char* kNuScenesCameras[6] = {
                "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                "CAM_BACK",  "CAM_BACK_LEFT",   "CAM_BACK_RIGHT"};
        return QString::fromLatin1(kNuScenesCameras[index]);
    }
    return QStringLiteral("Camera %1").arg(index);
}

// ImageNet channel statistics of the BEVFusion scene payload convention
// (upstream prepare_cameras normalizes with these before images.bin).
constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageNetStd[3] = {0.229f, 0.224f, 0.225f};

}  // namespace

bool readSceneCameras(const QString& sceneDir,
                      OpenPCDetSceneView* out,
                      QString* error) {
    auto fail = [error](const QString& msg) {
        if (error) *error = msg;
        return false;
    };
    if (out == nullptr) {
        return fail(QStringLiteral("Internal error: null output"));
    }
    *out = OpenPCDetSceneView{};
    const QFileInfo info(sceneDir);
    if (!info.isDir()) {
        return fail(QStringLiteral("Not a scene directory: %1").arg(sceneDir));
    }
    QFile json(QDir(sceneDir).filePath(QStringLiteral("scene.json")));
    if (!json.open(QIODevice::ReadOnly)) {
        return fail(QStringLiteral("Missing scene.json in %1").arg(sceneDir));
    }
    QJsonParseError parseError{};
    const QJsonDocument doc =
            QJsonDocument::fromJson(json.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        return fail(QStringLiteral("Malformed scene.json in %1").arg(sceneDir));
    }
    const QJsonObject obj = doc.object();
    const int numCameras = obj.value(QStringLiteral("num_cameras")).toInt();
    const int height = obj.value(QStringLiteral("image_height")).toInt();
    const int width = obj.value(QStringLiteral("image_width")).toInt();
    const int channels = obj.value(QStringLiteral("image_channels")).toInt();
    if (numCameras <= 0 || height <= 0 || width <= 0 || channels <= 0) {
        return fail(
                QStringLiteral("Incomplete scene.json in %1").arg(sceneDir));
    }
    const QDir dir(sceneDir);
    const auto readFloats = [&dir, &fail](const QString& name,
                                          qint64 expectedBytes,
                                          std::vector<float>* dst) {
        QFile file(dir.filePath(name));
        if (!file.open(QIODevice::ReadOnly)) {
            fail(QStringLiteral("Missing %1 in %2").arg(name, dir.path()));
            return false;
        }
        const QByteArray bytes = file.readAll();
        if (bytes.size() != expectedBytes) {
            fail(QStringLiteral("%1 has %2 B, expected %3 B in %4")
                         .arg(name)
                         .arg(bytes.size())
                         .arg(expectedBytes)
                         .arg(dir.path()));
            return false;
        }
        dst->resize(static_cast<size_t>(expectedBytes / sizeof(float)));
        std::memcpy(dst->data(), bytes.constData(),
                    static_cast<size_t>(expectedBytes));
        return true;
    };
    std::vector<float> images;
    std::vector<float> lidarToImage;
    std::vector<float> imageAug;
    std::vector<float> lidarAug;
    const qint64 frameBytes =
            static_cast<qint64>(height) * width * channels * sizeof(float);
    if (!readFloats(QStringLiteral("images.bin"), frameBytes * numCameras,
                    &images) ||
        !readFloats(QStringLiteral("lidar_to_image.bin"),
                    static_cast<qint64>(numCameras) * 16 * sizeof(float),
                    &lidarToImage) ||
        !readFloats(QStringLiteral("image_aug.bin"),
                    static_cast<qint64>(numCameras) * 16 * sizeof(float),
                    &imageAug) ||
        !readFloats(QStringLiteral("lidar_aug.bin"), 16 * sizeof(float),
                    &lidarAug)) {
        return false;
    }
    // Display normalization: the CaDDN payload stores 0..1 frames while
    // the BEVFusion payload stores ImageNet-normalized values (negative
    // minima / maxima above 1). Any out-of-range value identifies the
    // normalized convention deterministically; denormalize for display
    // only — the inference payload on disk is never touched.
    float minVal = images[0], maxVal = images[0];
    for (float v : images) {
        minVal = std::min(minVal, v);
        maxVal = std::max(maxVal, v);
    }
    if (minVal < 0.0f || maxVal > 1.0f) {
        if (channels < 3) {
            return fail(
                    QStringLiteral(
                            "Normalized scene payload needs RGB channels in %1")
                            .arg(sceneDir));
        }
        for (size_t px = 0; px < images.size() / channels; ++px) {
            for (int c = 0; c < 3; ++c) {
                float& v = images[px * channels + c];
                v = v * kImageNetStd[c] + kImageNetMean[c];
            }
        }
    }
    QStringList names;
    const QJsonArray nameArray =
            obj.value(QStringLiteral("camera_names")).toArray();
    for (const QJsonValue& value : nameArray) {
        if (value.isString()) names.append(value.toString());
    }
    out->cameras.reserve(numCameras);
    for (int c = 0; c < numCameras; ++c) {
        OpenPCDetSceneCamera camera;
        camera.image = sceneCameraPreview(images.data(), numCameras, height,
                                          width, channels, c);
        if (camera.image.isNull()) {
            return fail(QStringLiteral("Unsupported scene payload in %1")
                                .arg(sceneDir));
        }
        camera.name = names.size() == numCameras
                              ? names.at(c)
                              : defaultCameraName(c, numCameras);
        std::memcpy(camera.lidarToImage,
                    lidarToImage.data() + static_cast<size_t>(c) * 16,
                    16 * sizeof(float));
        std::memcpy(camera.imageAug,
                    imageAug.data() + static_cast<size_t>(c) * 16,
                    16 * sizeof(float));
        out->cameras.append(camera);
    }
    std::memcpy(out->lidarAug, lidarAug.data(), 16 * sizeof(float));
    return true;
}

QVector<QLineF> projectBoxEdges(const float* corners24,
                                const OpenPCDetSceneView& scene,
                                int camera) {
    QVector<QLineF> out;
    if (corners24 == nullptr || camera < 0 || camera >= scene.cameras.size()) {
        return out;
    }
    const OpenPCDetSceneCamera& cam = scene.cameras.at(camera);
    const int width = cam.image.width();
    const int height = cam.image.height();
    if (width <= 0 || height <= 0) return out;
    // Undo the lidar augmentation (engine depth_lss project_depth chain).
    double laInv[9];
    if (!mat3Inverse(scene.lidarAug, laInv)) return out;
    const double laT[3] = {scene.lidarAug[3], scene.lidarAug[7],
                           scene.lidarAug[11]};
    // Corners -> camera homogeneous space (lidar_to_image applied before
    // the perspective divide so near-plane clipping stays linear).
    double q[8][3];
    for (int i = 0; i < 8; ++i) {
        const double p[3] = {corners24[3 * i + 0] - laT[0],
                             corners24[3 * i + 1] - laT[1],
                             corners24[3 * i + 2] - laT[2]};
        const double u[3] = {
                laInv[0] * p[0] + laInv[1] * p[1] + laInv[2] * p[2],
                laInv[3] * p[0] + laInv[4] * p[1] + laInv[5] * p[2],
                laInv[6] * p[0] + laInv[7] * p[1] + laInv[8] * p[2]};
        const float* m = cam.lidarToImage;
        q[i][0] = m[0] * u[0] + m[1] * u[1] + m[2] * u[2] + m[3];
        q[i][1] = m[4] * u[0] + m[5] * u[1] + m[6] * u[2] + m[7];
        q[i][2] = m[8] * u[0] + m[9] * u[1] + m[10] * u[2] + m[11];
    }
    // Same near-plane threshold the engine skips points below.
    constexpr double nearZ = 1e-5;
    const float* ia = cam.imageAug;
    for (const EdgePair& edge : boxEdges()) {
        double a[3] = {q[edge.a][0], q[edge.a][1], q[edge.a][2]};
        double b[3] = {q[edge.b][0], q[edge.b][1], q[edge.b][2]};
        if (a[2] < nearZ && b[2] < nearZ) continue;
        if (a[2] < nearZ || b[2] < nearZ) {
            const double fraction = (nearZ - a[2]) / (b[2] - a[2]);
            if (a[2] < nearZ) {
                a[0] += fraction * (b[0] - a[0]);
                a[1] += fraction * (b[1] - a[1]);
                a[2] = nearZ;
            } else {
                b[0] += fraction * (a[0] - b[0]);
                b[1] += fraction * (a[1] - b[1]);
                b[2] = nearZ;
            }
        }
        // Perspective divide, then the image augmentation (resize/crop).
        const double ax = a[0] / a[2], ay = a[1] / a[2];
        const double bx = b[0] / b[2], by = b[1] / b[2];
        const QPointF start(ia[0] * ax + ia[1] * ay + ia[2] + ia[3],
                            ia[4] * ax + ia[5] * ay + ia[6] + ia[7]);
        const QPointF end(ia[0] * bx + ia[1] * by + ia[2] + ia[3],
                          ia[4] * bx + ia[5] * by + ia[6] + ia[7]);
        QLineF clipped;
        if (clipSegmentToRect(start, end, width, height, &clipped)) {
            out.append(clipped);
        }
    }
    return out;
}

void paintDetectionsOverlay(QImage* image,
                            const QVector<OpenPCDetDetection>& detections,
                            const OpenPCDetSceneView& scene,
                            int camera) {
    if (image == nullptr || image->isNull() || camera < 0 ||
        camera >= scene.cameras.size() || detections.isEmpty()) {
        return;
    }
    QImage canvas = image->convertToFormat(QImage::Format_ARGB32);
    QPainter painter(&canvas);
    painter.setRenderHint(QPainter::Antialiasing);
    QFont font = painter.font();
    font.setPixelSize(qMax(11, canvas.height() / 24));
    painter.setFont(font);
    const QFontMetrics metrics(font);
    for (const OpenPCDetDetection& det : detections) {
        float corners[24];
        boxCorners(det, corners);
        const QVector<QLineF> segments =
                projectBoxEdges(corners, scene, camera);
        if (segments.isEmpty()) continue;
        const OpenPCDetClassColor color = classColor(det.label, 0);
        const QColor penColor(color.r, color.g, color.b);
        painter.setPen(QPen(penColor, 2));
        painter.drawLines(segments);
        // Class + score label at the topmost projected corner, clamped
        // inside the frame; a 1px dark shadow keeps it readable on any
        // background.
        QPointF top = segments.first().p1();
        for (const QLineF& segment : segments) {
            if (segment.p1().y() < top.y()) top = segment.p1();
            if (segment.p2().y() < top.y()) top = segment.p2();
        }
        const QString name = det.className.isEmpty() ? className(det.label, 0)
                                                     : det.className;
        const QString text =
                QStringLiteral("%1 %2").arg(name).arg(det.score, 0, 'f', 2);
        const int textWidth = QTCOMPAT_FONTMETRICS_WIDTH(metrics, text);
        const int textHeight = metrics.height();
        const int x = qBound(0, static_cast<int>(top.x()),
                             qMax(0, canvas.width() - textWidth));
        const int y = qBound(textHeight, static_cast<int>(top.y()) - 4,
                             canvas.height());
        painter.setPen(QColor(0, 0, 0, 160));
        painter.drawText(x + 1, y + 1, text);
        painter.setPen(penColor);
        painter.drawText(x, y, text);
    }
    painter.end();
    *image = canvas.convertToFormat(QImage::Format_RGB888);
}

OpenPCDetTestDataFiles testDataFilesForCheckpoint(const QString& checkpoint,
                                                  const QString& frameId) {
    OpenPCDetTestDataFiles out;
    if (checkpoint.isEmpty()) return out;
    const bool scene =
            inputKindForFilename(checkpoint + QStringLiteral(".f16.gguf")) ==
            InputKind::Scene;
    const bool nuscenes = checkpoint.startsWith(QStringLiteral("nuscenes"));
    // Empty frameId keeps the deterministic first fill (the pinned default
    // frame of each dataset); the dialog's Frame combo passes the selected
    // frame of the SAME dataset (KITTI panels list KITTI frames only).
    QString frame = frameId;
    if (frame.isEmpty()) {
        frame = nuscenes ? QStringLiteral("nuscenes_n015")
                         : QStringLiteral("kitti_000008");
    }
    const QString frameDir = QStringLiteral("frames/%1/").arg(frame);
    const QString rawFile =
            frameDir + (nuscenes ? frame + QStringLiteral("_lidar.bin")
                                 : frame + QStringLiteral(".bin"));
    if (scene) {
        // CaDDN / focal VoxelRCNN (KITTI single-camera) and BEVFusion
        // (nuScenes 6-camera): the frame's scene/ directory + its raw
        // LiDAR frame as the LiDAR branch (the AICore scene pairing).
        out.sceneDirName = frameDir + QStringLiteral("scene");
        out.scenePointsFile = rawFile;
        return out;
    }
    if (!nuscenes && checkpoint.contains(QStringLiteral("pointrcnn"))) {
        // PointRCNN checkpoints train on the pre-sampled 16384-point
        // variant of the same frame.
        out.pointsFile = frameDir + frame + QStringLiteral("_sampled16384.bin");
    } else {
        out.pointsFile = rawFile;
    }
    return out;
}

bool resolveTestDataPaths(const QString& bundleRoot,
                          const QString& checkpoint,
                          const QString& frameId,
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
    const OpenPCDetTestDataFiles files =
            testDataFilesForCheckpoint(checkpoint, frameId);
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
