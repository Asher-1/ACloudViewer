// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qFreeSplatter.h"

#include <FileIOFilter.h>
#include <ecvCameraSensor.h>
#include <ecvCameraSensorDisplayUtils.h>
#include <ecvColorScalesManager.h>
#include <ecvColorTypes.h>
#include <ecvGLMatrix.h>
#include <ecvImage.h>
#include <ecvMainAppInterface.h>
#include <ecvNormalVectors.h>
#include <ecvPluginDbNaming.h>
#ifdef HAS_QSIBR
#include <ecvPluginManager.h>
#endif
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvViewManager.h>

#include <QAction>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMainWindow>
#include <QMessageBox>
#include <QSet>
#include <QSettings>
#include <QStandardPaths>
#include <QTimer>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/gaussian_capi.h"

namespace {

// Match ImageColormapUniform defaults in app/reconstruction (colormaps.cc).
constexpr ecvColor::Rgb kColmapCameraPlaneColor(255, 25, 0);
constexpr ecvColor::Rgb kColmapCameraFrameColor(204, 25, 0);
// COLMAP viewer frustum half-width ≈ this fraction of adaptive scene diameter.
constexpr float kCameraFrustumSceneExtentFraction = 0.03f;

constexpr int kMaxInferenceViews = 64;  // AICore ingest_images hard cap
constexpr int kSceneTargetViews = 2;    // scene GGUF recipe
constexpr int kObjectTargetViews = 16;  // object: practical multi-view cap
// ply_export always writes 45 f_rest coeffs (standard 3DGS SH degree 3 layout).
constexpr int kSibrPlyShDegree = 3;

struct SceneCameras {
    bool hasPoses = false;
    QVector<float> cam2world;
    float focal = 500.0f;
};

FreeSplatterDialog::ModelType modelTypeFromPath(const QString& modelPath) {
    if (modelPath.contains("scene", Qt::CaseInsensitive)) {
        return FreeSplatterDialog::ModelType::Scene;
    }
    if (modelPath.contains("object", Qt::CaseInsensitive)) {
        return FreeSplatterDialog::ModelType::Object;
    }
    return FreeSplatterDialog::ModelType::Unknown;
}

int inferenceViewCap(const QString& modelPath) {
    switch (modelTypeFromPath(modelPath)) {
        case FreeSplatterDialog::ModelType::Scene:
            return kSceneTargetViews;
        case FreeSplatterDialog::ModelType::Object:
            return kObjectTargetViews;
        default:
            return kMaxInferenceViews;
    }
}

QStringList uniformSubsamplePaths(const QStringList& paths, int targetCount) {
    if (targetCount <= 0 || paths.size() <= targetCount) {
        return paths;
    }
    QStringList out;
    out.reserve(targetCount);
    for (int i = 0; i < targetCount; ++i) {
        const int idx = static_cast<int>(
                (static_cast<int64_t>(i) * (paths.size() - 1)) /
                (targetCount - 1));
        out.append(paths[idx]);
    }
    return out;
}

// OpenCV-style cam2world: +X right, +Y down, +Z forward; columns are camera
// axes.
void buildLookAtCam2world(
        float ex, float ey, float ez, float cx, float cy, float cz, float* m) {
    float fx = cx - ex;
    float fy = cy - ey;
    float fz = cz - ez;
    const float flen = std::sqrt(fx * fx + fy * fy + fz * fz);
    if (flen > 1e-6f) {
        fx /= flen;
        fy /= flen;
        fz /= flen;
    } else {
        fx = 0.0f;
        fy = 0.0f;
        fz = 1.0f;
    }

    const float wx = 0.0f;
    const float wy = 1.0f;
    const float wz = 0.0f;

    float rx = fy * wz - fz * wy;
    float ry = fz * wx - fx * wz;
    float rz = fx * wy - fy * wx;
    float rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (rlen < 1e-6f) {
        rx = 1.0f;
        ry = 0.0f;
        rz = 0.0f;
    } else {
        rx /= rlen;
        ry /= rlen;
        rz /= rlen;
    }

    const float dx = fy * rz - fz * ry;
    const float dy = fz * rx - fx * rz;
    const float dz = fx * ry - fy * rx;

    m[0] = rx;
    m[1] = ry;
    m[2] = rz;
    m[4] = dx;
    m[5] = dy;
    m[6] = dz;
    m[8] = fx;
    m[9] = fy;
    m[10] = fz;
    m[3] = ex;
    m[7] = ey;
    m[11] = ez;
    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;
}

float sceneMaxExtentFromBBox(const ccBBox& bbox) {
    if (!bbox.isValid()) {
        return 1.0f;
    }
    const CCVector3 diag = bbox.maxCorner() - bbox.minCorner();
    const float dx = std::abs(diag.x);
    const float dy = std::abs(diag.y);
    const float dz = std::abs(diag.z);
    const float maxAxis = std::max({dx, dy, dz, 1e-4f});
    const float minAxis = std::max({std::min({dx, dy, dz}), 1e-4f});
    // Geometric mean adapts to flat vs cubic scenes better than max axis alone.
    const float geoDiameter =
            2.0f * std::sqrt(std::max(maxAxis * minAxis, 1e-8f));
    const float avgDiameter = (dx + dy + dz) / 3.0f;
    return std::max({geoDiameter, avgDiameter * 0.5f, minAxis, 1e-4f});
}

// Robust scene diameter from Gaussian centers (COLMAP world), ignoring
// outliers.
bool computeAdaptiveSceneExtent(const FreeSplatterResult& result,
                                float opacityThreshold,
                                float& outExtent) {
    if (result.gaussians.isEmpty() || result.nViews < 1 || result.height < 1 ||
        result.width < 1 || result.gaussianChannels < 16) {
        return false;
    }

    const int gc = result.gaussianChannels;
    const int64_t count =
            static_cast<int64_t>(result.nViews) * result.height * result.width;

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    int valid = 0;
    for (int64_t i = 0; i < count; ++i) {
        const float* g = result.gaussians.constData() + i * gc;
        if (g[15] <= opacityThreshold) continue;
        cx += g[0];
        cy += g[1];
        cz += g[2];
        ++valid;
    }
    if (valid == 0) {
        return false;
    }
    cx /= valid;
    cy /= valid;
    cz /= valid;

    std::vector<float> radii;
    radii.reserve(static_cast<size_t>(valid));
    for (int64_t i = 0; i < count; ++i) {
        const float* g = result.gaussians.constData() + i * gc;
        if (g[15] <= opacityThreshold) continue;
        const double dx = g[0] - cx;
        const double dy = g[1] - cy;
        const double dz = g[2] - cz;
        radii.push_back(
                static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz)));
    }
    if (radii.empty()) {
        return false;
    }

    std::sort(radii.begin(), radii.end());
    const size_t p90Index =
            static_cast<size_t>(0.9 * static_cast<double>(radii.size() - 1));
    const float p90Radius = radii[p90Index];
    outExtent = std::max(2.0f * p90Radius, 1e-4f);
    return true;
}

ccCameraSensor* buildCameraSensorFromPose(const float* rowMajorCam2world,
                                          float focalPx,
                                          int imageWidth,
                                          int imageHeight,
                                          float imageDisplaySize,
                                          float sceneMaxExtent) {
    if (!rowMajorCam2world || imageWidth < 1 || imageHeight < 1) {
        return nullptr;
    }

    // Same COLMAP/VtkColmap convention as ModelViewerWidget (native world
    // coords).
    const float display_focal_mm =
            ecvCameraSensorDisplay::ComputeFrustumDisplayFocalMm(
                    imageDisplaySize, imageWidth, imageHeight, focalPx);
    const float viewport_v_fov_rad =
            ecvCameraSensorDisplay::ComputeVerticalFovRad(focalPx, imageHeight);

    auto* sensor = new ccCameraSensor();
    sensor->setPoseFrame(ccCameraSensor::PoseFrame::VtkColmap);

    sensor->setPlaneColor(kColmapCameraPlaneColor);
    sensor->setFrameColor(kColmapCameraFrameColor);

    int retinaScale = 1;
    if (auto* view = ecvViewManager::instance().getEffectiveView()) {
        retinaScale = std::max(view->getDevicePixelRatio(), 1);
    }

    // FreeSplatter gaussians live in ~1-unit scenes — shrink graphic scale to
    // match.
    const float ar =
            static_cast<float>(imageWidth) / static_cast<float>(imageHeight);
    const float halfDisplayFov = viewport_v_fov_rad * 0.5f;
    const float frustumHalfWidthAtUnitScale =
            std::abs(display_focal_mm) * std::tan(halfDisplayFov * ar);
    const float targetHalfWidth =
            std::max(sceneMaxExtent, 1e-4f) * kCameraFrustumSceneExtentFraction;
    float graphicScaleFactor = 1.0f;
    if (frustumHalfWidthAtUnitScale > 1e-6f) {
        graphicScaleFactor = targetHalfWidth / frustumHalfWidthAtUnitScale;
    }

    ccCameraSensor::IntrinsicParameters iParams;
    iParams.zNear_mm = 1e-3f;
    const float estImagePlaneDepth =
            std::abs(display_focal_mm) * graphicScaleFactor;
    iParams.zFar_mm = std::max(
            std::max(sceneMaxExtent * 3.0f, estImagePlaneDepth * 4.0f), 1e-3f);
    iParams.pixelSize_mm[0] = imageDisplaySize;
    iParams.pixelSize_mm[1] = imageDisplaySize;
    iParams.vertFocal_pix = ccCameraSensor::ConvertFocalMMToPix(
            display_focal_mm, imageDisplaySize);
    iParams.vFOV_rad = viewport_v_fov_rad;
    iParams.arrayWidth = imageWidth;
    iParams.arrayHeight = imageHeight;
    iParams.principal_point[0] = static_cast<float>(imageWidth) * 0.5f;
    iParams.principal_point[1] = static_cast<float>(imageHeight) * 0.5f;
    sensor->setIntrinsicParameters(iParams);
    sensor->setApplyViewportVFov_rad(viewport_v_fov_rad);

    sensor->setGraphicScale(PC_ONE /
                            static_cast<PointCoordinateType>(retinaScale) *
                            graphicScaleFactor);

    sensor->setRigidTransformation(
            ecvCameraSensorDisplay::RowMajorCam2worldToVtkCameraSensorMatrix(
                    rowMajorCam2world));

    sensor->setEnabled(true);
    sensor->setVisible(true);
    sensor->setLocked(true);
    return sensor;
}

bool buildGaussianDisplayBBox(const FreeSplatterResult& result,
                              float opacityThreshold,
                              ccBBox& out) {
    if (result.gaussians.isEmpty() || result.nViews < 1 || result.height < 1 ||
        result.width < 1 || result.gaussianChannels < 16) {
        return false;
    }

    const int gc = result.gaussianChannels;
    const int64_t count =
            static_cast<int64_t>(result.nViews) * result.height * result.width;
    float minx = std::numeric_limits<float>::max();
    float miny = minx;
    float minz = minx;
    float maxx = -minx;
    float maxy = maxx;
    float maxz = maxx;
    int valid = 0;

    for (int64_t i = 0; i < count; ++i) {
        const float* g = result.gaussians.constData() + i * gc;
        if (g[15] <= opacityThreshold) continue;
        const float x = g[0];
        const float y = g[1];
        const float z = g[2];
        minx = std::min(minx, x);
        miny = std::min(miny, y);
        minz = std::min(minz, z);
        maxx = std::max(maxx, x);
        maxy = std::max(maxy, y);
        maxz = std::max(maxz, z);
        ++valid;
    }

    if (valid == 0) {
        return false;
    }

    out = ccBBox(CCVector3(minx, miny, minz), CCVector3(maxx, maxy, maxz));
    out.setValidity(true);
    return true;
}

bool computeGaussianBounds(const FreeSplatterResult& result,
                           float opacityThreshold,
                           float& cx,
                           float& cy,
                           float& cz,
                           float& radius) {
    if (result.gaussians.isEmpty() || result.nViews < 1 || result.height < 1 ||
        result.width < 1 || result.gaussianChannels < 16) {
        return false;
    }

    const int gc = result.gaussianChannels;
    const int64_t count =
            static_cast<int64_t>(result.nViews) * result.height * result.width;
    float minx = std::numeric_limits<float>::max();
    float miny = minx;
    float minz = minx;
    float maxx = -minx;
    float maxy = maxx;
    float maxz = maxx;
    int valid = 0;

    for (int64_t i = 0; i < count; ++i) {
        const float* g = result.gaussians.constData() + i * gc;
        if (g[15] <= opacityThreshold) continue;
        const float x = g[0];
        const float y = g[1];
        const float z = g[2];
        minx = std::min(minx, x);
        miny = std::min(miny, y);
        minz = std::min(minz, z);
        maxx = std::max(maxx, x);
        maxy = std::max(maxy, y);
        maxz = std::max(maxz, z);
        ++valid;
    }

    if (valid == 0) return false;

    cx = 0.5f * (minx + maxx);
    cy = 0.5f * (miny + maxy);
    cz = 0.5f * (minz + maxz);
    const float dx = maxx - minx;
    const float dy = maxy - miny;
    const float dz = maxz - minz;
    radius = 0.5f * std::sqrt(dx * dx + dy * dy + dz * dz);
    if (radius < 0.05f) radius = 1.0f;
    return true;
}

SceneCameras resolveSceneCameras(const FreeSplatterResult& result,
                                 float opacityThreshold,
                                 bool boundsOrbitOnly = false) {
    SceneCameras cameras;
    if (!boundsOrbitOnly && result.hasPoses &&
        result.cam2world.size() >= result.nViews * 16) {
        cameras.hasPoses = true;
        cameras.cam2world = result.cam2world;
        cameras.focal = result.focal > 0 ? result.focal : 500.0f;
        return cameras;
    }

    if (!boundsOrbitOnly && result.nViews >= 2 && !result.gaussians.isEmpty()) {
        cameras.cam2world.resize(result.nViews * 16);
        float focal = 0.0f;
        if (aicore_gaussian_estimate_poses(
                    result.gaussians.constData(), result.nViews, result.height,
                    result.width, result.gaussianChannels, opacityThreshold,
                    cameras.cam2world.data(), &focal) == 0) {
            cameras.hasPoses = true;
            cameras.focal = focal > 0 ? focal : 500.0f;
            return cameras;
        }
    }

    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    float radius = 1.0f;
    if (!computeGaussianBounds(result, opacityThreshold, cx, cy, cz, radius)) {
        return cameras;
    }

    const float dist = std::max(radius * 2.5f, 1.5f);
    const int nViews = std::max(result.nViews, 1);
    cameras.cam2world.resize(nViews * 16);
    cameras.focal = result.focal > 0 ? result.focal : 500.0f;
    cameras.hasPoses = true;

    for (int v = 0; v < nViews; ++v) {
        const float angle =
                (nViews > 1) ? (6.2831853f * static_cast<float>(v) / nViews)
                             : 0.0f;
        const float px = cx + dist * 0.35f * std::sin(angle);
        const float py = cy;
        const float pz = cz + dist * std::cos(angle);
        buildLookAtCam2world(px, py, pz, cx, cy, cz,
                             cameras.cam2world.data() + v * 16);
    }

    return cameras;
}

QByteArray sceneCamerasToJson(const FreeSplatterResult& result,
                              const SceneCameras& sceneCams) {
    if (!sceneCams.hasPoses || sceneCams.cam2world.isEmpty()) {
        return {};
    }

    const int W = result.width > 0 ? result.width : 512;
    const int H = result.height > 0 ? result.height : 512;
    const float focal = sceneCams.focal > 0 ? sceneCams.focal : 500.0f;
    const int nViews = std::max(result.nViews, 1);

    QJsonArray cameras;
    auto addCamera = [&](int id, const QString& name, float px, float py,
                         float pz, const float rot[3][3]) {
        QJsonObject cam;
        cam["id"] = id;
        cam["img_name"] = name;
        cam["width"] = W;
        cam["height"] = H;
        cam["fx"] = static_cast<double>(focal);
        cam["fy"] = static_cast<double>(focal);
        cam["position"] = QJsonArray{px, py, pz};
        QJsonArray rotRows;
        for (int r = 0; r < 3; ++r) {
            rotRows.append(QJsonArray{rot[r][0], rot[r][1], rot[r][2]});
        }
        cam["rotation"] = rotRows;
        cameras.append(cam);
    };

    for (int v = 0; v < nViews; ++v) {
        const float* m = sceneCams.cam2world.constData() + v * 16;
        const float rot[3][3] = {
                {m[0], m[1], m[2]}, {m[4], m[5], m[6]}, {m[8], m[9], m[10]}};
        addCamera(v, QString("view%1.jpg").arg(v), m[3], m[7], m[11], rot);
    }

    return QJsonDocument(cameras).toJson(QJsonDocument::Compact);
}

QString gaussianChannelScalarName(int channel) {
    if (channel >= 3 && channel <= 14) {
        return QStringLiteral("SH_%1").arg(channel - 3);
    }
    switch (channel) {
        case 15:
            return QStringLiteral("Opacity");
        case 16:
            return QStringLiteral("Scale_0");
        case 17:
            return QStringLiteral("Scale_1");
        case 18:
            return QStringLiteral("Scale_2");
        default:
            return QStringLiteral("Ch_%1").arg(channel);
    }
}

// ccPointCloud per-point slots: XYZ, RGB, compressed normals, scalar fields.
// Quaternion components are not meaningful as separate scalar fields; full 3DGS
// rotation belongs in SIBR PLY, not standard point-cloud export.
bool shouldExportGaussianAsScalarField(int channel) {
    if (channel == 15) return true;                    // Opacity
    if (channel >= 3 && channel <= 14) return true;    // SH coefficients
    if (channel >= 16 && channel <= 18) return true;   // anisotropic scale
    if (channel >= 19 && channel <= 22) return false;  // quaternion — skip
    return channel > 22;
}

void applyOpacityGreyColorScale(ccScalarField* opacitySF) {
    if (!opacitySF) return;
    auto* scales = ccColorScalesManager::GetUniqueInstance();
    if (!scales) return;
    opacitySF->setColorScale(
            scales->getDefaultScale(ccColorScalesManager::GREY));
}

// Same quaternion -> rotation matrix mapping as SIBR CudaRasterizer
// (forward.cu).
void rotationMatrixFromQuaternionWXYZ(
        float w, float x, float y, float z, float R[3][3]) {
    R[0][0] = 1.f - 2.f * (y * y + z * z);
    R[0][1] = 2.f * (x * y - w * z);
    R[0][2] = 2.f * (x * z + w * y);
    R[1][0] = 2.f * (x * y + w * z);
    R[1][1] = 1.f - 2.f * (x * x + z * z);
    R[1][2] = 2.f * (y * z - w * x);
    R[2][0] = 2.f * (x * z - w * y);
    R[2][1] = 2.f * (y * z + w * x);
    R[2][2] = 1.f - 2.f * (x * x + y * y);
}

int minScaleAxisIndex(float s0, float s1, float s2) {
    if (s1 <= s0 && s1 <= s2) return 1;
    if (s2 <= s0 && s2 <= s1) return 2;
    return 0;
}

// Thin-axis direction of the Gaussian ellipsoid (M = S*R), COLMAP world.
bool computeMinScaleAxisNormal(const float* p, int gc, CCVector3& outNormal) {
    if (gc < 23) return false;

    const float s0 = p[16];
    const float s1 = p[17];
    const float s2 = p[18];
    const float minS = std::min({s0, s1, s2});
    const float maxS = std::max({s0, s1, s2});
    constexpr float kMinAnisotropyRatio = 1.2f;
    if (minS <= 0.0f || maxS / minS < kMinAnisotropyRatio) return false;

    float w = p[19];
    float x = p[20];
    float y = p[21];
    float z = p[22];
    const float len2 = w * w + x * x + y * y + z * z;
    if (len2 < 1e-12f) return false;
    const float invLen = 1.0f / std::sqrt(len2);
    w *= invLen;
    x *= invLen;
    y *= invLen;
    z *= invLen;

    float R[3][3];
    rotationMatrixFromQuaternionWXYZ(w, x, y, z, R);

    const int axis = minScaleAxisIndex(s0, s1, s2);
    float nx = R[0][axis];
    float ny = R[1][axis];
    float nz = R[2][axis];

    const float nLen = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (nLen < 1e-8f) return false;

    outNormal = CCVector3(nx / nLen, ny / nLen, nz / nLen);
    return true;
}

QString freesplatterModelKindTag(const QString& modelPath) {
    const QString filename = QFileInfo(modelPath).fileName();
    if (filename.contains(QStringLiteral("scene"), Qt::CaseInsensitive)) {
        return QStringLiteral("Scene");
    }
    if (filename.contains(QStringLiteral("object"), Qt::CaseInsensitive)) {
        return QStringLiteral("Object");
    }
    return ecvPluginDbNaming::modelTagFromFilename(modelPath);
}

QString buildGaussianExportBaseName(
        const FreeSplatterResult& result,
        const FreeSplatterDialog::Settings& settings) {
    const QString modelKind = freesplatterModelKindTag(settings.modelPath);
    const QString modelQuant =
            ecvPluginDbNaming::modelTagFromFilename(settings.modelPath, 24);

    QString source = ecvPluginDbNaming::sanitizeSegment(result.sourceName);
    if (source.isEmpty()) source = QStringLiteral("run");

    QString name =
            QStringLiteral("FS_%1_%2_%3").arg(modelKind, modelQuant, source);

    if (result.nViews > 0) {
        name += QStringLiteral("_%1v").arg(result.nViews);
    }
    if (settings.exportFieldMode == FreeSplatterDialog::ExportFieldMode::Full) {
        name += QStringLiteral("_Full");
    }
    return name;
}

}  // namespace

#ifdef HAS_QSIBR
static constexpr const char kQSIBRPluginName[] = "SIBR Viewer";
#endif

qFreeSplatter::qFreeSplatter(QObject* parent)
    : QObject(parent),
      ccStdPluginInterface(":/CC/plugin/qFreeSplatter/info.json") {
    qRegisterMetaType<FreeSplatterResult>("FreeSplatterResult");
    qRegisterMetaType<FreeSplatterDialog::Settings>(
            "FreeSplatterDialog::Settings");
    m_action = new QAction("FreeSplatter 3D Reconstruction", this);
    m_action->setToolTip(
            "FreeSplatter — turn photos into 3D Gaussians without camera "
            "poses");
    m_action->setIcon(
            QIcon(":/CC/plugin/qFreeSplatter/images/qFreeSplatter.svg"));
    connect(m_action, &QAction::triggered, this, &qFreeSplatter::showDialog);

    m_inferenceHeartbeat = new QTimer(this);
    m_inferenceHeartbeat->setInterval(10000);
    connect(m_inferenceHeartbeat, &QTimer::timeout, this, [this]() {
        if (!m_worker || !m_worker->isRunning() || !m_dialog) return;
        m_inferenceElapsedSeconds += 10;
        m_dialog->appendLog(
                tr("[FS] Task is running (%1 s elapsed)...")
                        .arg(m_inferenceElapsedSeconds));
    });
}

void qFreeSplatter::onNewSelection(
        const ccHObject::Container& selectedEntities) {
    m_selectedEntities = selectedEntities;
    if (!m_dialog || !m_dialog->isVisible()) return;

    QStringList imageNames;
    for (ccHObject* obj : selectedEntities) {
        if (obj && obj->isA(CV_TYPES::IMAGE)) {
            imageNames.append(obj->getName());
        }
    }
    if (!imageNames.isEmpty()) {
        m_dialog->applyDbTreeSelection(imageNames);
    }
}

QList<QAction*> qFreeSplatter::getActions() { return {m_action}; }

void qFreeSplatter::refreshDbImages() {
    if (!m_app || !m_dialog) return;
    ccHObject* root = m_app->dbRootObject();
    if (!root) {
        m_dialog->setDbImages({});
        return;
    }
    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);

    QList<FreeSplatterDialog::DbImageEntry> entries;
    for (ccHObject* obj : images) {
        if (!obj || !obj->isEnabled()) continue;
        ccImage* img = dynamic_cast<ccImage*>(obj);
        if (!img) continue;
        FreeSplatterDialog::DbImageEntry entry;
        entry.name = obj->getName();
        entry.preview = img->data();
        entries.append(entry);
    }
    m_dialog->setDbImages(entries);
}

ccImage* qFreeSplatter::findDbImage(const QString& name) const {
    if (!m_app) return nullptr;
    ccHObject* root = m_app->dbRootObject();
    if (!root) return nullptr;

    ccHObject::Container images;
    root->filterChildren(images, true, CV_TYPES::IMAGE, false);
    for (ccHObject* obj : images) {
        if (obj && obj->getName() == name) {
            return dynamic_cast<ccImage*>(obj);
        }
    }
    return nullptr;
}

QStringList qFreeSplatter::selectedDbImageNames() const {
    QStringList names;
    for (ccHObject* obj : m_selectedEntities) {
        if (obj && obj->isA(CV_TYPES::IMAGE)) {
            names.append(obj->getName());
        }
    }
    return names;
}

bool qFreeSplatter::resolveInputPaths(const QStringList& rawPaths,
                                      QStringList& outPaths,
                                      QString* errorMsg) const {
    outPaths.clear();
    const QString tmpDir = FreeSplatterDialog::modelCacheDir() + "/../tmp";
    QDir().mkpath(tmpDir);

    for (const QString& raw : rawPaths) {
        if (raw.startsWith("db://")) {
            const QString name = raw.mid(5);
            ccImage* img = findDbImage(name);
            if (!img || img->data().isNull()) {
                if (errorMsg) {
                    *errorMsg = tr("DB image not found or empty: %1").arg(name);
                }
                return false;
            }
            const QString safeName = name;
            const QString tmpPath = tmpDir + "/" + safeName + ".png";
            if (!img->data().save(tmpPath)) {
                if (errorMsg) {
                    *errorMsg = tr("Failed to export DB image: %1").arg(name);
                }
                return false;
            }
            outPaths << tmpPath;
        } else if (QFile::exists(raw)) {
            outPaths << raw;
        } else {
            if (errorMsg) {
                *errorMsg = tr("Input file not found: %1").arg(raw);
            }
            return false;
        }
    }
    return true;
}

void qFreeSplatter::onWorkerProgress(int current, int total) {
    if (!m_dialog) return;
    m_dialog->setProgress(current, total);
    QString stage;
    if (current < 15) {
        stage = tr("Loading GGUF model...");
    } else if (current < 25) {
        stage = tr("Preparing inference...");
    } else if (current < 75) {
        stage = tr("Running 3D Gaussian inference... (%1%)").arg(current);
    } else if (current < 100) {
        stage = tr("Building result for DB display...");
    } else {
        stage = tr("Done.");
    }
    m_dialog->setTaskStage(stage, current);
}

QByteArray qFreeSplatter::buildSibrCamerasJson(const FreeSplatterResult& result,
                                               float opacityThreshold) {
#ifdef HAS_QSIBR
    if (result.gaussians.isEmpty() || result.nViews < 1) return {};

    // Orbit cameras from scene bounds — robust initial framing for SIBR.
    const SceneCameras sceneCams =
            resolveSceneCameras(result, opacityThreshold, true);
    if (!sceneCams.hasPoses) return {};
    return sceneCamerasToJson(result, sceneCams);
#else
    (void)result;
    (void)opacityThreshold;
    return {};
#endif
}

bool qFreeSplatter::warmupInferenceBackend(const QString& device,
                                           QString* logMsg) const {
#ifdef AICore_ENABLED
    const QByteArray dev =
            device.isEmpty() ? QByteArray("auto") : device.toUtf8();
    if (aicore_gaussian_warmup_backend(dev.constData()) != 0) {
        if (logMsg) {
            *logMsg =
                    tr("[Warning] GPU backend warmup failed — worker will "
                       "retry (try CPU if it crashes).");
        }
        return false;
    }
    return true;
#else
    (void)device;
    (void)logMsg;
    return false;
#endif
}

bool qFreeSplatter::launchSibrGaussianViewer(const QByteArray& plyBytes,
                                             const QByteArray& camerasJson,
                                             int shDegree) {
#ifdef HAS_QSIBR
    if (plyBytes.isEmpty() || camerasJson.isEmpty()) return false;

    for (ccPluginInterface* plugin : ccPluginManager::get().pluginList()) {
        if (!plugin || plugin->getName() != QLatin1String(kQSIBRPluginName)) {
            continue;
        }
        auto* sibrObj = dynamic_cast<QObject*>(plugin);
        if (!sibrObj) return false;

        return QMetaObject::invokeMethod(
                sibrObj, "launchInMemoryGaussianViewer", Qt::QueuedConnection,
                Q_ARG(QByteArray, plyBytes), Q_ARG(QByteArray, camerasJson),
                Q_ARG(int, shDegree));
    }
    return false;
#else
    (void)plyBytes;
    (void)camerasJson;
    (void)shDegree;
    return false;
#endif
}

void qFreeSplatter::showDialog() {
    if (!m_app) return;
    if (!m_dialog) {
        m_dialog = new FreeSplatterDialog(m_app->getMainWindow());
        connect(m_dialog, &FreeSplatterDialog::runRequested, this,
                &qFreeSplatter::executeTask);
        connect(m_dialog, &FreeSplatterDialog::cancelRequested, this,
                &qFreeSplatter::cancelTask);
#ifdef HAS_QSIBR
        connect(m_dialog, &FreeSplatterDialog::visualizeRequested, this,
                &qFreeSplatter::onVisualizeRequested);
#endif
        connect(m_dialog, &FreeSplatterDialog::exportPlyRequested, this,
                &qFreeSplatter::onExportPlyRequested);
        connect(m_dialog, &FreeSplatterDialog::refreshDbImagesRequested, this,
                [this]() { refreshDbImages(); });
    }
    refreshDbImages();
    const QStringList selectedNames = selectedDbImageNames();
    if (!selectedNames.isEmpty()) {
        m_dialog->applyDbTreeSelection(selectedNames);
    }
    m_dialog->show();
    m_dialog->raise();
    m_dialog->activateWindow();
}

void qFreeSplatter::executeTask(const FreeSplatterDialog::Settings& settings) {
    if (m_worker && m_worker->isRunning()) {
        QMessageBox::warning(m_dialog, "FreeSplatter",
                             "A task is already running.");
        return;
    }
    if (m_worker) {
        m_worker->disconnect(this);
        m_worker->disconnect(m_dialog);
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    m_lastResult = {};
    m_lastPlyBytes.clear();
#ifdef HAS_QSIBR
    m_lastSibrCamerasJson.clear();
#endif
    m_lastDbCloud = nullptr;
    m_lastDbCameraGroup = nullptr;
    if (m_dialog) {
        m_dialog->enableResultButtons(false);
    }

    FreeSplatterDialog::Settings resolvedSettings = settings;

    if (resolvedSettings.mode == FreeSplatterDialog::Mode::Reconstruct &&
        !resolvedSettings.inputPaths.isEmpty()) {
        QStringList resolvedPaths;
        QString err;
        if (!resolveInputPaths(resolvedSettings.inputPaths, resolvedPaths,
                               &err)) {
            m_dialog->appendLog("[Error] " + err);
            return;
        }
        resolvedSettings.inputPaths = resolvedPaths;
        m_dialog->appendLog(tr("[FS] Resolved %1 input image(s).")
                                    .arg(resolvedPaths.size()));

        const int originalCount = resolvedSettings.inputPaths.size();
        const int viewCap =
                std::min(kMaxInferenceViews,
                         inferenceViewCap(resolvedSettings.modelPath));
        if (originalCount > viewCap) {
            resolvedSettings.inputPaths =
                    uniformSubsamplePaths(resolvedSettings.inputPaths, viewCap);
            m_dialog->appendLog(
                    tr("[FS] %1 input images exceed model limit — uniformly "
                       "subsampled to %2 (cap %3, hard max %4).")
                            .arg(originalCount)
                            .arg(resolvedSettings.inputPaths.size())
                            .arg(viewCap)
                            .arg(kMaxInferenceViews));
        }
    }

    if (resolvedSettings.modelPath.isEmpty()) {
        m_dialog->appendLog("[Error] Please select a GGUF model file.");
        return;
    }

    if (resolvedSettings.mode == FreeSplatterDialog::Mode::Reconstruct &&
        resolvedSettings.inputPaths.isEmpty()) {
        m_dialog->appendLog(
                "[Error] No input images. Use File/Folder, check DB images, "
                "or select ccImage entities in the DB tree.");
        return;
    }

    m_currentSettings = resolvedSettings;

#ifdef HAS_QSIBR
    m_dialog->appendLog(tr(
            "[FS] Tip: close the SIBR viewer before GPU inference if the GPU "
            "is already in use."));
#endif

    QString warmupMsg;
    QString workerDevice = resolvedSettings.device;
    if (!warmupInferenceBackend(resolvedSettings.device, &warmupMsg)) {
        if (!warmupMsg.isEmpty()) {
            m_dialog->appendLog(warmupMsg);
        }
        if (aicore_is_gpu_device(workerDevice.toUtf8().constData())) {
            workerDevice = QStringLiteral("cpu");
            m_dialog->appendLog(tr(
                    "[FS] GPU backend unavailable — using CPU for this run."));
        }
    } else {
        m_dialog->appendLog(tr("[FS] Inference backend ready on UI thread."));
    }

    FreeSplatterWorker::Settings workerSettings;
    workerSettings.mode = static_cast<FreeSplatterWorker::Mode>(
            static_cast<int>(resolvedSettings.mode));
    workerSettings.modelPath = resolvedSettings.modelPath;
    workerSettings.inputPaths = resolvedSettings.inputPaths;
    workerSettings.threads = resolvedSettings.threads;
    workerSettings.device = workerDevice;
    workerSettings.opacityThreshold = resolvedSettings.opacityThreshold;
    workerSettings.estimatePoses = resolvedSettings.estimatePoses;

    m_worker = new FreeSplatterWorker(workerSettings, this);
    connect(m_worker, &FreeSplatterWorker::logMessage, m_dialog,
            &FreeSplatterDialog::appendLog, Qt::QueuedConnection);
    connect(m_worker, &FreeSplatterWorker::progressUpdate, this,
            &qFreeSplatter::onWorkerProgress, Qt::QueuedConnection);
    connect(m_worker, &FreeSplatterWorker::resultReady, this,
            &qFreeSplatter::onResultReady, Qt::QueuedConnection);
    connect(m_worker, &FreeSplatterWorker::modelInfoReady, this,
            &qFreeSplatter::onModelInfo, Qt::QueuedConnection);
    connect(m_worker, &FreeSplatterWorker::taskFinished, this,
            &qFreeSplatter::onTaskFinished, Qt::QueuedConnection);

    m_dialog->setRunning(true);
    m_dialog->appendLog("[FS] Starting task...");
    m_inferenceElapsedSeconds = 0;
    m_inferenceHeartbeat->start();
    m_worker->start();
}

void qFreeSplatter::cancelTask() {
    if (m_worker && m_worker->isRunning()) {
        m_worker->requestInterruption();
        m_dialog->appendLog("[FS] Cancel requested...");
    }
}

ccPointCloud* qFreeSplatter::buildResultPointCloud(
        const FreeSplatterResult& result,
        float opacityThreshold,
        FreeSplatterDialog::ExportFieldMode exportFieldMode,
        const QString& cloudName,
        int* validCountOut) const {
    const int N = result.nViews * result.height * result.width;
    const int gc = result.gaussianChannels;
    const float* g = result.gaussians.constData();
    if (gc < 16) return nullptr;

    int validCount = 0;
    for (int i = 0; i < N; ++i) {
        if (g[i * gc + 15] > opacityThreshold) validCount++;
    }
    if (validCountOut) *validCountOut = validCount;
    if (validCount == 0) return nullptr;

    auto* cloud = new ccPointCloud(
            cloudName.isEmpty() ? QStringLiteral("FS_Gaussians") : cloudName);
    if (!cloud->reserve(static_cast<unsigned>(validCount))) {
        delete cloud;
        if (validCountOut) *validCountOut = 0;
        return nullptr;
    }

    const bool exportFull =
            exportFieldMode == FreeSplatterDialog::ExportFieldMode::Full;

    QVector<int> extraChannels;
    if (exportFull) {
        for (int ch = 3; ch < gc; ++ch) {
            if (ch == 15 || !shouldExportGaussianAsScalarField(ch)) continue;
            extraChannels.append(ch);
        }
    }

    QHash<int, ccScalarField*> scalarFields;
    auto* opacitySF = new ccScalarField("Opacity");
    const bool hasOpacitySF =
            opacitySF->reserveSafe(static_cast<unsigned>(validCount));
    if (hasOpacitySF) {
        applyOpacityGreyColorScale(opacitySF);
        scalarFields.insert(15, opacitySF);
    } else {
        opacitySF->release();
    }

    for (int ch : extraChannels) {
        auto* sf = new ccScalarField(qPrintable(gaussianChannelScalarName(ch)));
        if (sf->reserveSafe(static_cast<unsigned>(validCount))) {
            scalarFields.insert(ch, sf);
        } else {
            sf->release();
        }
    }

    const bool hasColors = cloud->reserveTheRGBTable();
    const bool exportNormals = exportFull && gc >= 23;
    const float C0 = 0.28209479177387814f;
    const CompressedNormType defaultNormalIndex =
            ccNormalVectors::GetNormIndex(CCVector3(0.0f, 0.0f, 1.0f));

    for (int i = 0; i < N; ++i) {
        const float* p = g + i * gc;
        if (p[15] <= opacityThreshold) continue;

        cloud->addPoint(CCVector3(static_cast<PointCoordinateType>(p[0]),
                                  static_cast<PointCoordinateType>(p[1]),
                                  static_cast<PointCoordinateType>(p[2])));

        if (hasColors) {
            auto r = static_cast<ColorCompType>(
                    std::clamp(0.5f + C0 * p[3], 0.0f, 1.0f) * 255.0f);
            auto gC = static_cast<ColorCompType>(
                    std::clamp(0.5f + C0 * p[4], 0.0f, 1.0f) * 255.0f);
            auto b = static_cast<ColorCompType>(
                    std::clamp(0.5f + C0 * p[5], 0.0f, 1.0f) * 255.0f);
            cloud->addRGBColor(r, gC, b);
        }

        if (exportNormals) {
            CCVector3 normal;
            if (computeMinScaleAxisNormal(p, gc, normal)) {
                cloud->addNormIndex(ccNormalVectors::GetNormIndex(normal));
            } else {
                cloud->addNormIndex(defaultNormalIndex);
            }
        }

        for (auto it = scalarFields.constBegin(); it != scalarFields.constEnd();
             ++it) {
            it.value()->addElement(static_cast<ScalarType>(p[it.key()]));
        }
    }

    for (ccScalarField* sf : scalarFields.values()) {
        sf->computeMinAndMax();
        cloud->addScalarField(sf);
    }

    if (hasColors) cloud->showColors(true);
    cloud->setVisible(true);
    cloud->setEnabled(true);
    return cloud;
}

void qFreeSplatter::addResultToDb(const FreeSplatterResult& result) {
    if (!m_app) return;

    const QString baseName =
            buildGaussianExportBaseName(result, m_currentSettings);
    const QString cloudName = ecvPluginDbNaming::makeUnique(baseName, m_app);

    int validCount = 0;
    ccPointCloud* cloud = buildResultPointCloud(
            result, m_currentSettings.opacityThreshold,
            m_currentSettings.exportFieldMode, cloudName, &validCount);
    if (!cloud || validCount == 0) {
        m_dialog->appendLog("[Warning] No valid gaussians after pruning.");
        delete cloud;
        return;
    }

    m_app->addToDB(cloud, true, true, false, true);
    m_app->setSelectedInDB(cloud, true);
    m_app->zoomOnEntities(cloud);
    m_app->refreshAll();

    m_lastDbCloud = cloud;

    const bool fullExport = m_currentSettings.exportFieldMode ==
                            FreeSplatterDialog::ExportFieldMode::Full;
    m_dialog->appendLog(
            QString("[FS] Added '%1' to DB tree (%2 gaussians, %3 export)")
                    .arg(cloud->getName())
                    .arg(validCount)
                    .arg(fullExport ? tr("full attributes") : tr("basic")));

    if (m_currentSettings.estimatePoses) {
        addCameraPosesToDb(result, baseName, cloud);
    }
}

void qFreeSplatter::addCameraPosesToDb(const FreeSplatterResult& result,
                                       const QString& baseName,
                                       ccPointCloud* cloud) {
    if (!m_app) return;
    if (!result.hasPoses || result.cam2world.size() < result.nViews * 16 ||
        result.nViews < 1) {
        m_dialog->appendLog(
                tr("[FS] Camera poses: enabled but unavailable — skipped DB "
                   "camera "
                   "export."));
        return;
    }

    const QString folderName = ecvPluginDbNaming::makeUnique(
            baseName + QStringLiteral("_Cameras"), m_app);
    auto* group = new ccHObject(folderName);
    group->setEnabled(true);

    const int W = result.width > 0 ? result.width : 512;
    const int H = result.height > 0 ? result.height : 512;
    const float focal = result.focal > 0 ? result.focal : 500.0f;
    float imageDisplaySize = 1024.0f;
    if (auto* view = ecvViewManager::instance().getEffectiveView()) {
        imageDisplaySize *=
                static_cast<float>(std::max(view->getDevicePixelRatio(), 1));
    }
    ccBBox sceneBoundsStorage;
    float sceneMaxExtent = 1.0f;
    if (!computeAdaptiveSceneExtent(result, m_currentSettings.opacityThreshold,
                                    sceneMaxExtent)) {
        if (buildGaussianDisplayBBox(result, m_currentSettings.opacityThreshold,
                                     sceneBoundsStorage)) {
            sceneMaxExtent = sceneMaxExtentFromBBox(sceneBoundsStorage);
        } else if (cloud) {
            sceneBoundsStorage = cloud->getOwnBB();
            if (sceneBoundsStorage.isValid()) {
                sceneMaxExtent = sceneMaxExtentFromBBox(sceneBoundsStorage);
            }
        }
    }

    for (int v = 0; v < result.nViews; ++v) {
        QString viewName;
        if (v < m_currentSettings.inputPaths.size()) {
            const QString& path = m_currentSettings.inputPaths[v];
            viewName = path.startsWith(QStringLiteral("db://"))
                               ? path.mid(5)
                               : QFileInfo(path).completeBaseName();
        }
        if (viewName.isEmpty()) {
            viewName = QStringLiteral("view%1").arg(v);
        }
        viewName = ecvPluginDbNaming::sanitizeSegment(viewName, 32);

        ccCameraSensor* sensor = buildCameraSensorFromPose(
                result.cam2world.constData() + v * 16, focal, W, H,
                imageDisplaySize, sceneMaxExtent);
        if (!sensor) continue;
        sensor->setName(viewName);
        group->addChild(sensor);
    }

    if (group->getChildrenNumber() == 0) {
        delete group;
        m_dialog->appendLog(tr(
                "[FS] Camera poses: failed to build ccCameraSensor entities."));
        return;
    }

    m_app->addToDB(group, true, true, false, true);
    m_lastDbCameraGroup = group;
    m_dialog->appendLog(tr("[FS] Added '%1' to DB tree (%2 ccCameraSensor, "
                           "COLMAP-compatible "
                           "frustum)")
                                .arg(folderName)
                                .arg(group->getChildrenNumber()));
}

void qFreeSplatter::onResultReady(const FreeSplatterResult& result) {
    if (!m_app) return;

    m_lastResult = result;

    if (m_currentSettings.addToDb) {
        addResultToDb(result);
    } else {
        m_dialog->appendLog("[FS] Skipped DB export (disabled in settings).");
    }

    const int gc = result.gaussianChannels;
    const float* g = result.gaussians.constData();
    const float opThr = m_currentSettings.opacityThreshold;

    m_lastPlyBytes.clear();
    unsigned char* plyBytes = nullptr;
    size_t plySize = 0;
    const int shDegree = result.shDegree > 0 ? result.shDegree : 1;
    if (aicore_gaussian_export_ply_bytes(g, result.nViews, result.height,
                                         result.width, gc, shDegree, opThr,
                                         &plyBytes, &plySize) == 0 &&
        plyBytes && plySize > 0) {
        m_lastPlyBytes = QByteArray(reinterpret_cast<const char*>(plyBytes),
                                    static_cast<int>(plySize));
        aicore_gaussian_free_bytes(plyBytes);
    }

    const bool hasResult = !result.gaussians.isEmpty();
    if (hasResult) {
        m_dialog->enableResultButtons(true);
    }
#ifdef HAS_QSIBR
    const bool canVisualize = !m_lastPlyBytes.isEmpty();
    m_lastSibrCamerasJson =
            hasResult ? buildSibrCamerasJson(result,
                                             m_currentSettings.opacityThreshold)
                      : QByteArray();
    if (!m_lastSibrCamerasJson.isEmpty() && canVisualize) {
        m_dialog->appendLog(tr("[FS] SIBR payload ready in memory (PLY %1 "
                               "bytes, cameras %2 bytes)")
                                    .arg(m_lastPlyBytes.size())
                                    .arg(m_lastSibrCamerasJson.size()));
    } else if (canVisualize) {
        m_dialog->appendLog(
                tr("[Warning] Failed to prepare in-memory SIBR cameras."));
    }
#endif

    if (m_currentSettings.estimatePoses) {
        if (result.hasPoses) {
            m_dialog->appendLog(
                    tr("[FS] Camera poses: estimated (%1 views, focal=%2 px). "
                       "Exported to DB as ccCameraSensor folder when Add to DB "
                       "is "
                       "enabled; PLY export writes *_cameras.json sidecar.")
                            .arg(result.nViews)
                            .arg(result.focal, 0, 'f', 2));
        } else {
            m_dialog->appendLog(
                    tr("[FS] Camera poses: estimation enabled but failed "
                       "(non-fatal). No camera folder in DB."));
        }
    } else {
        m_dialog->appendLog(tr(
                "[FS] Camera poses: disabled — no ccCameraSensor DB export."));
    }
}

void qFreeSplatter::onModelInfo(const QString& info) {
    m_dialog->appendLog("[FS] Model info:\n" + info);
}

void qFreeSplatter::onTaskFinished(bool success) {
    if (m_inferenceHeartbeat) m_inferenceHeartbeat->stop();
    if (success) {
        m_dialog->appendLog("[FS] Task finished.");
        m_dialog->setProgress(100, 100);
        m_dialog->setTaskStage(tr("Done."), 100);
    } else {
        m_dialog->appendLog("[Error] Task failed — see log above.");
        m_dialog->setProgress(0, 100);
        m_dialog->setTaskStage(tr("Failed."), 0);
        m_lastResult = {};
        m_lastPlyBytes.clear();
#ifdef HAS_QSIBR
        m_lastSibrCamerasJson.clear();
#endif
        if (m_dialog) {
            m_dialog->enableResultButtons(false);
        }
    }
    m_dialog->setRunning(false);
    if (success) {
        m_dialog->enableResultButtons(!m_lastResult.gaussians.isEmpty());
    }
    if (m_worker) {
        m_worker->releaseContextOnMainThread();
        m_worker->deleteLater();
        m_worker = nullptr;
    }

    const QString tmpDir = FreeSplatterDialog::modelCacheDir() + "/../tmp";
    QDir tmp(tmpDir);
    if (tmp.exists()) {
        tmp.removeRecursively();
    }

    if (m_app) {
        m_app->updateUI();
        m_app->refreshAll();
    }
}

#ifdef HAS_QSIBR
void qFreeSplatter::onVisualizeRequested() {
    if (m_lastPlyBytes.isEmpty()) {
        QMessageBox::information(
                m_dialog, "FreeSplatter",
                "No Gaussian result available for visualization.");
        return;
    }

    QByteArray camerasJson = m_lastSibrCamerasJson;
    if (camerasJson.isEmpty()) {
        camerasJson = buildSibrCamerasJson(m_lastResult,
                                           m_currentSettings.opacityThreshold);
    }
    if (camerasJson.isEmpty()) {
        m_dialog->appendLog(
                tr("[Error] Failed to prepare in-memory SIBR cameras."));
        QMessageBox::warning(m_dialog, "FreeSplatter",
                             tr("Could not prepare SIBR viewer cameras."));
        return;
    }

    if (!launchSibrGaussianViewer(m_lastPlyBytes, camerasJson,
                                  kSibrPlyShDegree)) {
        m_dialog->appendLog(
                tr("[Error] SIBR plugin unavailable — enable "
                   "PLUGIN_STANDARD_QSIBR."));
        QMessageBox::warning(
                m_dialog, "FreeSplatter",
                tr("Could not launch SIBR viewer (qSIBR plugin not loaded)."));
        return;
    }

    m_dialog->appendLog(tr("[FS] Launching SIBR Gaussian viewer (in-memory PLY "
                           "%1 bytes, cameras %2 bytes)")
                                .arg(m_lastPlyBytes.size())
                                .arg(camerasJson.size()));
}
#endif

void qFreeSplatter::onExportPlyRequested() {
    if (m_lastResult.gaussians.isEmpty()) {
        QMessageBox::information(m_dialog, "FreeSplatter",
                                 "No result available to export.");
        return;
    }

    const FreeSplatterDialog::Settings exportSettings =
            m_dialog ? m_dialog->getSettings() : m_currentSettings;

    int validCount = 0;
    const QString baseName =
            buildGaussianExportBaseName(m_lastResult, exportSettings);
    std::unique_ptr<ccPointCloud> cloud(buildResultPointCloud(
            m_lastResult, exportSettings.opacityThreshold,
            exportSettings.exportFieldMode, baseName, &validCount));
    if (!cloud || validCount == 0) {
        QMessageBox::warning(m_dialog, "FreeSplatter",
                             "No valid gaussians to export.");
        return;
    }

    const QString defaultName = baseName + QStringLiteral(".ply");
    QSettings settings;
    const QString lastExportDir =
            settings.value("qFreeSplatter/lastExportDir",
                           QStandardPaths::writableLocation(
                                   QStandardPaths::DocumentsLocation))
                    .toString();
    const QString defaultPath =
            lastExportDir.isEmpty() ? defaultName
                                    : QDir(lastExportDir).filePath(defaultName);
    const QString path = QFileDialog::getSaveFileName(
            m_dialog, tr("Export Point Cloud (PLY)"), defaultPath,
            tr("PLY files (*.ply)"));
    if (path.isEmpty()) return;

    settings.setValue("qFreeSplatter/lastExportDir",
                      QFileInfo(path).absolutePath());

    FileIOFilter::SaveParameters saveParams;
    saveParams.alwaysDisplaySaveDialog = false;
    saveParams.parentWidget = m_dialog;

    CC_FILE_ERROR err = CC_FERR_UNKNOWN_FILE;
    if (auto filter = FileIOFilter::FindBestFilterForExtension("ply")) {
        err = FileIOFilter::SaveToFile(cloud.get(), path, saveParams, filter);
    }
    if (err != CC_FERR_NO_ERROR) {
        FileIOFilter::DisplayErrorMessage(err, tr("saving"), path);
        m_dialog->appendLog(tr("[Error] PLY export failed: %1").arg(path));
        return;
    }

    m_dialog->appendLog(tr("[FS] PLY exported via CV_io: %1 (%2 points)")
                                .arg(path)
                                .arg(cloud->size()));

    if (!m_lastPlyBytes.isEmpty()) {
        const QString gsPlyPath = QFileInfo(path).absolutePath() + "/" +
                                  QFileInfo(path).completeBaseName() +
                                  "_3dgs.ply";
        QFile gsFile(gsPlyPath);
        if (gsFile.open(QIODevice::WriteOnly)) {
            gsFile.write(m_lastPlyBytes);
            gsFile.close();
            m_dialog->appendLog(tr("[FS] 3DGS PLY exported (SIBR-compatible): "
                                   "%1 (%2 bytes)")
                                        .arg(gsPlyPath)
                                        .arg(m_lastPlyBytes.size()));
        }
    }

    const bool wantPoseJson =
            exportSettings.estimatePoses && m_lastResult.hasPoses;
    const SceneCameras poseCams =
            wantPoseJson ? resolveSceneCameras(m_lastResult,
                                               exportSettings.opacityThreshold,
                                               false)
                         : SceneCameras{};
    if (poseCams.hasPoses) {
        const QString camPath = QFileInfo(path).absolutePath() + "/" +
                                QFileInfo(path).completeBaseName() +
                                "_cameras.json";
        const QByteArray poseJson = sceneCamerasToJson(m_lastResult, poseCams);
        QFile camFile(camPath);
        if (!poseJson.isEmpty() && camFile.open(QIODevice::WriteOnly)) {
            camFile.write(poseJson);
            camFile.close();
            m_dialog->appendLog(tr("[FS] Estimated camera poses exported: %1")
                                        .arg(camPath));
        }
    }
}
