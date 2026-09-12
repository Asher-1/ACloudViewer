// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TrellisWorker.h"

#include <QCryptographicHash>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPainter>
#include <QPolygon>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "TrellisModelCatalog.h"
#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/trellis_capi.h"

namespace {

/* Serializes this worker against every other AICore inference task on the
 * same device (live video loops, other plugin workers). ggml-metal's backend
 * state machine is not safe under concurrent graph compute from two threads;
 * the shared device queue lock is the process-wide mutex that keeps command
 * buffers from racing (a failed command buffer poisons the backend for the
 * rest of the process). */
class DeviceTaskGuard {
public:
    explicit DeviceTaskGuard(const QString& device)
        : m_locked(aicore_device_task_lock(device.toUtf8().constData()) == 0) {}
    ~DeviceTaskGuard() {
        if (m_locked) aicore_device_task_unlock();
    }
    bool isLocked() const { return m_locked; }

private:
    bool m_locked = false;
};

// Stage names for the progress log (mirror aicore_trellis_stage).
const char* stageName(int stage) {
    switch (stage) {
        case AICORE_TRELLIS_STAGE_PREPROCESS:
            return "preprocess";
        case AICORE_TRELLIS_STAGE_DINO:
            return "dino";
        case AICORE_TRELLIS_STAGE_SS_FLOW:
            return "ss_flow";
        case AICORE_TRELLIS_STAGE_SS_DEC:
            return "ss_dec";
        case AICORE_TRELLIS_STAGE_SLAT_FLOW:
            return "slat_flow";
        case AICORE_TRELLIS_STAGE_SHAPE_DEC:
            return "shape_dec";
        case AICORE_TRELLIS_STAGE_MESH:
            return "mesh";
        case AICORE_TRELLIS_STAGE_UPSAMPLE:
            return "upsample";
        case AICORE_TRELLIS_STAGE_SLAT_FLOW_HR:
            return "slat_flow_hr";
        case AICORE_TRELLIS_STAGE_SHAPE_DEC_HR:
            return "shape_dec_hr";
        case AICORE_TRELLIS_STAGE_TEXTURE:
            return "texture";
        default:
            return "?";
    }
}

// ── preview blob → QImage software rendering ────────────────────────────────
// The AICore preview blobs are self-describing little-endian payloads:
//   "T2VOX01": magic[8], u32 res, u32 nvox, u16[3*nvox] coords in [0,res)
//   "T2MESH01": magic[8], u32 nv, u32 nt, f32[3nv] verts, f32[3nv] normals,
//               i32[3nt] tris
// Both are rendered on the worker thread so the GUI thread only blits a
// ready QImage.

// ── thumbnail view selection ────────────────────────────────────────────────
// Both mesh renderers used to project straight onto world XY (camera on +Z);
// any mesh whose tall axis is Z collapsed into a horizontal sliver (bird
// models read as one thin line, see the pipeline-strip screenshots). A fixed
// "3/4 view" would hard-code an up convention TRELLIS meshes do not
// guarantee, so the basis is picked per mesh: try diagonal camera directions
// favouring each axis in turn and keep the first whose projected bounding box
// is not a sliver (min extent / max extent >= 0.15). Deterministic, cheap
// (one O(nv) extent pass per candidate), convention-agnostic.
struct ViewBasis {
    float r[3], u[3], d[3];  // screen right / screen up / toward camera
    float px_min = 0, px_max = 0, py_min = 0, py_max = 0;
};

static void basisFromDir(const float dir[3], const float up[3], ViewBasis& b) {
    float d[3], r[3], u[3];
    const float dl =
            std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int i = 0; i < 3; ++i) d[i] = dir[i] / dl;
    // right = normalize(up x d)
    r[0] = up[1] * d[2] - up[2] * d[1];
    r[1] = up[2] * d[0] - up[0] * d[2];
    r[2] = up[0] * d[1] - up[1] * d[0];
    const float rl =
            std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) + 1e-12f;
    for (int i = 0; i < 3; ++i) r[i] /= rl;
    // screen up = d x r (unit: d is perpendicular to r)
    u[0] = d[1] * r[2] - d[2] * r[1];
    u[1] = d[2] * r[0] - d[0] * r[2];
    u[2] = d[0] * r[1] - d[1] * r[0];
    for (int i = 0; i < 3; ++i) {
        b.d[i] = d[i];
        b.r[i] = r[i];
        b.u[i] = u[i];
    }
}

static ViewBasis pickViewBasis(const float* verts, size_t nv) {
    struct Candidate {
        float dir[3];
        float up[3];
    };
    static const Candidate kCandidates[] = {
            {{0.55f, 0.65f, 0.52f}, {0, 1, 0}},  // favours a Y-up mesh
            {{0.55f, 0.52f, 0.65f}, {0, 0, 1}},  // favours a Z-up mesh
            {{0.65f, 0.52f, 0.55f}, {1, 0, 0}},  // favours an X-up mesh
    };
    ViewBasis best{};
    float best_score = -1.0f;
    for (const Candidate& cand : kCandidates) {
        ViewBasis b;
        basisFromDir(cand.dir, cand.up, b);
        float px_min = 1e9f, px_max = -1e9f, py_min = 1e9f, py_max = -1e9f;
        for (size_t i = 0; i < nv; ++i) {
            const float* v = verts + i * 3;
            const float px = v[0] * b.r[0] + v[1] * b.r[1] + v[2] * b.r[2];
            const float py = v[0] * b.u[0] + v[1] * b.u[1] + v[2] * b.u[2];
            px_min = std::min(px_min, px);
            px_max = std::max(px_max, px);
            py_min = std::min(py_min, py);
            py_max = std::max(py_max, py);
        }
        const float ex = px_max - px_min, ey = py_max - py_min;
        const float score = std::min(ex, ey) / (std::max(ex, ey) + 1e-9f);
        if (score > best_score) {
            best = b;
            best.px_min = px_min;
            best.px_max = px_max;
            best.py_min = py_min;
            best.py_max = py_max;
            best_score = score;
        }
        if (score >= 0.15f) break;  // not a sliver — good enough
    }
    return best;
}

// Fit the projected bbox into (size - 8) px, centred, view-up -> screen up.
static void fitTransform(
        const ViewBasis& b, int size, float& scale, float& ox, float& oy) {
    const float ex = b.px_max - b.px_min, ey = b.py_max - b.py_min;
    scale = (float)(size - 8) / std::max(std::max(ex, ey), 1e-6f);
    ox = 4.0f + ((float)size - 8 - ex * scale) * 0.5f - b.px_min * scale;
    oy = 4.0f + ((float)size - 8 - ey * scale) * 0.5f + b.py_max * scale;
}
static void viewProject(const ViewBasis& b, const float* v, float* out) {
    out[0] = v[0] * b.r[0] + v[1] * b.r[1] + v[2] * b.r[2];
    out[1] = v[0] * b.u[0] + v[1] * b.u[1] + v[2] * b.u[2];
    out[2] = v[0] * b.d[0] + v[1] * b.d[1] + v[2] * b.d[2];
}

// Isometric voxel-set render (painter-sorted shaded cubes).
QImage renderVoxelBlob(const char* data, int len, int size) {
    if (len < 16) return QImage();
    quint32 res = 0, nvox = 0;
    std::memcpy(&res, data + 8, 4);
    std::memcpy(&nvox, data + 12, 4);
    if (res == 0 || res > 4096 || nvox == 0) return QImage();
    if (len < (int)(16 + (size_t)nvox * 6)) return QImage();
    const unsigned char* cells = (const unsigned char*)data + 16;
    const float r = (float)res;

    // Isometric axes: +x -> (0.866, 0.5), +z -> (-0.866, 0.5), +y -> (0, -1).
    auto project = [](float x, float y, float z) {
        return std::pair<float, float>((x - z) * 0.8660254f,
                                       (x + z) * 0.5f - y);
    };
    // Unit-cube bounds in projected space.
    float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
    const float corners[8][3] = {{0, 0, 0}, {r, 0, 0}, {0, r, 0}, {r, r, 0},
                                 {0, 0, r}, {r, 0, r}, {0, r, r}, {r, r, r}};
    for (const auto& c : corners) {
        auto p = project(c[0], c[1], c[2]);
        umin = std::min(umin, p.first);
        umax = std::max(umax, p.first);
        vmin = std::min(vmin, p.second);
        vmax = std::max(vmax, p.second);
    }
    const float span = std::max(umax - umin, vmax - vmin) + 1e-6f;
    const float scale = (float)(size - 8) / span;
    auto toScreen = [&](float u, float v) {
        return QPointF(4.0 + (u - umin) * scale, 4.0 + (vmax - v) * scale);
    };

    QImage img(size, size, QImage::Format_ARGB32_Premultiplied);
    img.fill(Qt::transparent);
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing, false);
    p.setPen(Qt::NoPen);

    // Depth sort: the camera looks along -(1,1,1), so voxels with smaller
    // x+y+z are farther away and are drawn first.
    struct Cell {
        quint16 x, y, z;
    };
    std::vector<Cell> order;
    order.reserve(nvox);
    for (quint32 i = 0; i < nvox; ++i) {
        Cell c;
        std::memcpy(&c.x, cells + (size_t)i * 6 + 0, 2);
        std::memcpy(&c.y, cells + (size_t)i * 6 + 2, 2);
        std::memcpy(&c.z, cells + (size_t)i * 6 + 4, 2);
        order.push_back(c);
    }
    std::sort(order.begin(), order.end(), [](const Cell& a, const Cell& b) {
        return (a.x + a.y + a.z) < (b.x + b.y + b.z);
    });

    const QColor top(126, 174, 224), side(72, 118, 168), front(52, 92, 138);
    for (const Cell& c : order) {
        const float x = c.x, y = c.y, z = c.z;
        auto v = [&](float dx, float dy, float dz) {
            auto pr = project(x + dx, y + dy, z + dz);
            return toScreen(pr.first, pr.second);
        };
        // Top face at y+1.
        QPointF topQ[4] = {v(0, 1, 0), v(1, 1, 0), v(1, 1, 1), v(0, 1, 1)};
        p.setBrush(top);
        p.drawPolygon(topQ, 4);
        // +x face.
        QPointF rightQ[4] = {v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1)};
        p.setBrush(side);
        p.drawPolygon(rightQ, 4);
        // +z face.
        QPointF frontQ[4] = {v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)};
        p.setBrush(front);
        p.drawPolygon(frontQ, 4);
    }
    p.end();
    return img;
}

// Flat-shaded z-buffer render of a mesh keyframe.
QImage renderMeshBlob(const char* data, int len, int size) {
    if (len < 16) return QImage();
    quint32 nv = 0, nt = 0;
    std::memcpy(&nv, data + 8, 4);
    std::memcpy(&nt, data + 12, 4);
    if (nv == 0 || nt == 0 || nv > (1u << 22) || nt > (1u << 22)) {
        return QImage();
    }
    const size_t want = 16 + (size_t)nv * 24 + (size_t)nt * 12;
    if (len < (int)want) return QImage();
    const float* verts = (const float*)(data + 16);
    const float* normals = verts + (size_t)nv * 3;
    const qint32* tris = (const qint32*)(normals + (size_t)nv * 3);

    QImage img(size, size, QImage::Format_ARGB32);
    img.fill(0x00000000);
    std::vector<float> depth((size_t)size * size, -1e9f);
    const ViewBasis basis = pickViewBasis(verts, nv);
    float scale = 1.0f, ox = 0.0f, oy = 0.0f;
    fitTransform(basis, size, scale, ox, oy);
    // View-space positions: raster + face lighting below then work in camera
    // space, so the shading stays consistent for any mesh orientation.
    std::vector<float> vp((size_t)nv * 3);
    for (quint32 i = 0; i < nv; ++i)
        viewProject(basis, verts + (size_t)i * 3, vp.data() + (size_t)i * 3);
    auto toScreen = [&](const float* v, float* sx, float* sy) {
        *sx = ox + v[0] * scale;
        *sy = oy - v[1] * scale;  // view up -> image top-down
    };
    // Simple directional light in view space.
    const float lx = 0.4f, ly = 0.7f, lz = 0.6f;
    for (quint32 t = 0; t < nt; ++t) {
        const qint32 ia = tris[3 * t + 0], ib = tris[3 * t + 1],
                     ic = tris[3 * t + 2];
        if (ia < 0 || ib < 0 || ic < 0 || ia >= (qint32)nv ||
            ib >= (qint32)nv || ic >= (qint32)nv)
            continue;
        const float* a = vp.data() + (size_t)ia * 3;
        const float* b = vp.data() + (size_t)ib * 3;
        const float* c = vp.data() + (size_t)ic * 3;
        float ax, ay, bx, by, cx, cy;
        toScreen(a, &ax, &ay);
        toScreen(b, &bx, &by);
        toScreen(c, &cx, &cy);
        const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        if (std::fabs(area) < 1e-9f) continue;
        const float nz =
                (b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0]);
        const float ny =
                (b[2] - a[2]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[2] - a[2]);
        const float n1 =
                (b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1]);
        const float nl = std::sqrt(nz * nz + ny * ny + n1 * n1) + 1e-20f;
        float shade = std::fabs((nz * lx + ny * ly + n1 * lz) / nl);
        shade = 0.25f + 0.75f * shade;
        const int R = (int)(232 * shade), G = (int)(166 * shade),
                  B = (int)(98 * shade);
        const QRgb col = qRgb(R, G, B);
        const int x0 = std::max(0, (int)std::floor(std::min({ax, bx, cx})));
        const int x1 =
                std::min(size - 1, (int)std::ceil(std::max({ax, bx, cx})));
        const int y0 = std::max(0, (int)std::floor(std::min({ay, by, cy})));
        const int y1 =
                std::min(size - 1, (int)std::ceil(std::max({ay, by, cy})));
        const float inv_area = 1.0f / area;
        const float cz = a[2] + b[2] + c[2];
        for (int py = y0; py <= y1; ++py) {
            for (int px = x0; px <= x1; ++px) {
                const float sx = px + 0.5f, sy = py + 0.5f;
                const float w0 =
                        ((bx - sx) * (cy - sy) - (by - sy) * (cx - sx)) *
                        inv_area;
                const float w1 =
                        ((cx - sx) * (ay - sy) - (cy - sy) * (ax - sx)) *
                        inv_area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < 0 || w1 < 0 || w2 < 0) continue;
                // View depth = w0*a[2]+w1*b[2]+w2*c[2] along +z (camera at +z).
                const float z = w0 * a[2] + w1 * b[2] + w2 * c[2] + cz * 0.0f;
                float& d = depth[(size_t)py * size + px];
                if (z > d) {
                    d = z;
                    img.setPixel(px, py, col);
                }
            }
        }
    }
    return img;
}

// Shaded vertex-splat render of the final mesh. Lambert-shaded with the
// per-vertex PBR base colour when textured, the same amber as the mesh
// keyframes otherwise. One z-tested splat per vertex: the uniformly
// distributed surface vertices cover the 256 px thumbnail gap-free, while a
// full triangle raster of a multi-million-triangle mesh would cost seconds.
QImage renderResultPreview(const QVector<float>& verts,
                           const QVector<float>& normals,
                           const QVector<float>& pbr,
                           int size) {
    const int nv = verts.size() / 3;
    if (nv <= 0 || size <= 0) return QImage();
    QImage img(size, size, QImage::Format_ARGB32);
    img.fill(0x00000000);
    std::vector<float> depth((size_t)size * size, -1e9f);
    const ViewBasis basis = pickViewBasis(verts.constData(), nv);
    float scale = 1.0f, ox = 0.0f, oy = 0.0f;
    fitTransform(basis, size, scale, ox, oy);
    // Light in view space (normals are rotated below), so the shading is
    // stable no matter which way the mesh faces.
    const float lx = 0.4f, ly = 0.7f, lz = 0.6f;
    const bool textured = pbr.size() >= nv * 6;
    const bool shaded = normals.size() >= nv * 3;
    for (int i = 0; i < nv; ++i) {
        const float* v = verts.constData() + i * 3;
        float pv[3];
        viewProject(basis, v, pv);
        const int px = int(ox + pv[0] * scale);
        const int py = int(oy - pv[1] * scale);
        if (px < 0 || py < 0 || px >= size || py >= size) continue;
        float shade = 1.0f;
        if (shaded) {
            const float* n = normals.constData() + i * 3;
            const float nx =
                    n[0] * basis.r[0] + n[1] * basis.r[1] + n[2] * basis.r[2];
            const float ny =
                    n[0] * basis.u[0] + n[1] * basis.u[1] + n[2] * basis.u[2];
            const float nz =
                    n[0] * basis.d[0] + n[1] * basis.d[1] + n[2] * basis.d[2];
            const float nl = std::sqrt(nx * nx + ny * ny + nz * nz) + 1e-20f;
            // Two-sided lambert: the structure-tensor normals may point away
            // from the view; |dot| avoids black patches either way.
            shade = 0.25f +
                    0.75f * std::fabs((nx * lx + ny * ly + nz * lz) / nl);
        }
        float R, G, B;
        if (textured) {
            const float* c = pbr.constData() + i * 6;
            R = std::min(1.0f, c[0]) * shade;
            G = std::min(1.0f, c[1]) * shade;
            B = std::min(1.0f, c[2]) * shade;
        } else {
            R = 0.910f * shade;
            G = 0.651f * shade;
            B = 0.384f * shade;
        }
        float& d = depth[(size_t)py * size + px];
        if (pv[2] > d) {
            d = v[2];
            img.setPixel(px, py,
                         qRgba(int(R * 255.0f), int(G * 255.0f),
                               int(B * 255.0f), 255));
        }
    }
    return img;
}

}  // namespace

QImage TrellisWorker::renderPreviewBlob(const char* data, int len, int size) {
    if (!data || len < 16) return QImage();
    if (std::memcmp(data, "T2VOX01", 7) == 0) {
        return renderVoxelBlob(data, len, size);
    }
    if (std::memcmp(data, "T2MESH01", 8) == 0) {
        return renderMeshBlob(data, len, size);
    }
    return QImage();
}

TrellisWorker::TrellisWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {}

TrellisWorker::~TrellisWorker() {
    // Never free m_ctx from this destructor: QThread objects are destroyed on
    // the thread that owns them (usually the GUI thread), while the AICore
    // context belongs to the worker thread. Freeing it here would race with
    // run() and crash. run() always releases m_ctx before it returns; this
    // destructor only joins a still-running thread so no context leaks.
    if (isRunning()) {
        requestInterruption();
        wait(5000);
    }
}

void TrellisWorker::run() {
    // QThread entry: an exception escaping here terminates the whole
    // process (qTerminate). The AICore C ABI fences its own exceptions;
    // this catch-all is the last-resort guard for this worker thread.
#ifdef AICore_ENABLED
    emit logMessage(
            QStringLiteral("[TRELLIS] Run mode: %1")
                    .arg(m_settings.runMode == TrellisRunMode::BakeOnly
                                 ? QStringLiteral("bake-only (re-bake from "
                                                  "the last result)")
                                 : (m_settings.runMode ==
                                                    TrellisRunMode::GeometryOnly
                                            ? QStringLiteral(
                                                      "geometry-only "
                                                      "(inference, no GLB "
                                                      "bake)")
                                            : QStringLiteral(
                                                      "full (inference + GLB "
                                                      "bake)"))));
    bool ok = false;
    try {
        ok = m_settings.runMode == TrellisRunMode::BakeOnly ? runBakeOnly()
                                                            : runInference();
    } catch (const std::exception& e) {
        emit logMessage(QStringLiteral("[TRELLIS] Unexpected failure: %1")
                                .arg(QString::fromUtf8(e.what())));
    } catch (...) {
        emit logMessage(QStringLiteral(
                "[TRELLIS] Unexpected failure (unknown exception)."));
    }
    emit taskFinished(ok);
#else
    emit logMessage(QStringLiteral("[TRELLIS] AICore not enabled."));
    emit taskFinished(false);
#endif
}

#ifdef AICore_ENABLED

void TrellisWorker::applySettingsToOptions(aicore_trellis_options* opts) {
    aicore_trellis_options_set_device(opts,
                                      m_settings.device.toUtf8().constData());
    aicore_trellis_options_set_threads(opts, m_settings.threads);
    aicore_trellis_options_set_shape_dec_placement(
            opts, m_settings.shapeDecPlacement.toUtf8().constData());
    if (m_settings.useRmbg && !m_settings.rmbgModelPath.isEmpty()) {
        aicore_trellis_options_set_rmbg_gguf(
                opts, m_settings.rmbgModelPath.toUtf8().constData());
    }
}

void TrellisWorker::emitPreviewBlob(
        int stage, int step, int total, const void* data, int len) {
    TrellisStagePreview preview;
    preview.stage = stage;
    preview.step = step;
    preview.total = total;
    preview.label = QString::fromLatin1(stageName(stage));
    if (total > 0)
        preview.label += QStringLiteral(" %1/%2").arg(step).arg(total);
    if (len > 8 && std::memcmp(data, "T2VOX01", 7) == 0)
        preview.rawBlob = QByteArray((const char*)data, len);
    preview.image = renderPreviewBlob((const char*)data, len, 256);
    if (!preview.image.isNull()) {
        emit stagePreview(preview);
    }
}

void TrellisWorker::onProgress(int stage, int step, int total) {
    // Per-stage wall-time buckets: the timer restarts only on a stage change,
    // so each bucket accumulates that stage's full duration (progress fires
    // at stage entry; the trailing stage closes after generate returns).
    if (m_lastStage != stage) {
        if (m_lastStage >= 0) {
            m_stageMs[m_lastStage] +=
                    static_cast<double>(m_stageTimer.elapsed());
        }
        m_lastStage = stage;
        m_stageTimer.restart();
    }
    emit progressUpdate(stage, step, total);
}

bool TrellisWorker::resolveRmbgModel() {
    // RMBG is a shared AICore task dependency. Its physical cache belongs to
    // qRMBG, not to the TRELLIS pipeline weights.
    const QString cacheDir = TrellisHelpers::rmbgModelCacheDir();
    for (const QString& name :
         {QStringLiteral("rmbg_q8.gguf"), QStringLiteral("rmbg_f16.gguf")}) {
        const QString path = cacheDir + QLatin1Char('/') + name;
        if (QFile::exists(path)) {
            m_settings.rmbgModelPath = path;
            return true;
        }
    }
    emit logMessage(QStringLiteral("[TRELLIS] RMBG model not found in %1 "
                                   "(download rmbg_q8.gguf / rmbg_f16.gguf "
                                   "first, e.g. via the qRMBG plugin). Falling "
                                   "back to solid-color background removal.")
                            .arg(cacheDir));
    m_settings.useRmbg = false;
    return true;
}

bool TrellisWorker::runInference() {
    DeviceTaskGuard taskGuard(m_settings.device);
    if (!taskGuard.isLocked()) {
        emit logMessage(
                tr("[TRELLIS] Failed to acquire the inference device; "
                   "another task is running."));
        return false;
    }
    if (m_settings.modelPaths.size() < 3) {
        emit logMessage(QStringLiteral("[TRELLIS] Model set incomplete."));
        return false;
    }

    // Load the image bytes up-front so the pipeline can be released right
    // after generation without keeping the file open.
    QFile file(m_settings.inputPath);
    if (!file.open(QIODevice::ReadOnly)) {
        emit logMessage(QStringLiteral("[TRELLIS] Cannot open image: %1")
                                .arg(m_settings.inputPath));
        return false;
    }
    const QByteArray imageBytes = file.readAll();
    file.close();

    // Keep the UTF-8 payloads alive for the whole call: QString::toUtf8()
    // returns a temporary whose .constData() would dangle as soon as the
    // expression ends, and aicore_trellis_load_opts reads these pointers
    // *after* loading the RMBG model (a long-running call).
    //
    // Index contract: m_settings.modelPaths MUST be in
    // aicore_trellis_model_paths field order (dino, ss_flow, ss_dec, slat_flow,
    // slat_hr_flow, shape_dec, shape_enc, tex_dec, tex_flow, tex_flow_hr).
    // Omitted fields stay as empty strings (the C API treats "" like NULL:
    // "omit this model"). The presets in TrellisModelCatalog.cpp guarantee this
    // ordering.
    QByteArray utf8Paths[10];
    auto keepAlive = [&](int i) -> const char* {
        utf8Paths[i] = m_settings.modelPaths.value(i).toUtf8();
        return utf8Paths[i].constData();
    };

    aicore_trellis_model_paths paths{};
    paths.dino_gguf = keepAlive(0);
    paths.ss_flow_gguf = keepAlive(1);
    paths.ss_dec_gguf = keepAlive(2);
    // Optional models: 512 (+3), 1024 (+4), PBR (+5..+8).
    const int n = m_settings.modelPaths.size();
    if (n > 3) paths.slat_flow_gguf = keepAlive(3);
    if (n > 4) paths.slat_hr_flow_gguf = keepAlive(4);
    if (n > 5) paths.shape_dec_gguf = keepAlive(5);
    if (m_settings.textureEnabled) {
        if (n > 6) paths.shape_enc_gguf = keepAlive(6);
        if (n > 7) paths.tex_dec_gguf = keepAlive(7);
        if (n > 8) paths.tex_flow_gguf = keepAlive(8);
        if (n > 9) paths.tex_flow_hr_gguf = keepAlive(9);
    }

    if (m_settings.useRmbg) resolveRmbgModel();

    // Same keep-alive requirement for the RMBG path: the options struct
    // copies the string immediately, but keep it unambiguous anyway.
    QByteArray rmbgUtf8;
    if (m_settings.useRmbg && !m_settings.rmbgModelPath.isEmpty()) {
        rmbgUtf8 = m_settings.rmbgModelPath.toUtf8();
    }

    aicore_trellis_options* opts = aicore_trellis_options_new();
    applySettingsToOptions(opts);
    if (m_settings.useRmbg && !rmbgUtf8.isEmpty()) {
        aicore_trellis_options_set_rmbg_gguf(opts, rmbgUtf8.constData());
    }
    aicore_trellis_ctx* ctx = aicore_trellis_load_opts(&paths, opts);
    aicore_trellis_options_free(opts);
    if (!ctx) {
        emit logMessage(QStringLiteral(
                "[TRELLIS] Model load failed (check that every preset file "
                "exists and is a valid GGUF)."));
        return false;
    }
    m_ctx = ctx;
    // Surface a backend downgrade (e.g. VRAM too small for the requested
    // GPU) so the user knows why the run fell back to another device.
    const char* note = aicore_trellis_backend_note(ctx);
    if (note && note[0]) {
        emit logMessage(
                QStringLiteral("[TRELLIS] %1").arg(QString::fromUtf8(note)));
    }

    aicore_trellis_generate_params params{};
    params.pipeline_type = m_settings.pipelineType;
    params.background_mode = m_settings.backgroundMode;
    params.seed = m_settings.seed;
    params.steps = m_settings.steps;
    params.guidance = static_cast<float>(m_settings.guidance);
    params.texture_steps = m_settings.textureSteps;
    // Live per-step voxel previews (stride auto: ~4 across the SS run) and
    // two replayed shape-flow mesh keyframes; the preview callback is only
    // wired when the dialog asked for live previews.
    params.preview_stride = m_settings.livePreview ? 0 : -1;
    params.keyframes = m_settings.livePreview ? 2 : 0;

    char err[512] = {0};
    QElapsedTimer timer;
    m_stageTimer.start();
    m_lastStage = -1;
    m_stageMs.clear();
    timer.start();
    aicore_trellis_mesh* mesh = nullptr;
    {
        // Fixed function pointers: a capture-less lambda decays to a plain
        // function pointer, but the two branches of a conditional expression
        // (lambda vs nullptr) do not share a type — bind through variables
        // with explicit types so the live-preview switch stays type-safe.
        aicore_trellis_progress_fn progressFn = [](void* user, int stage,
                                                   int step, int total) {
            auto* self = static_cast<TrellisWorker*>(user);
            self->onProgress(stage, step, total);
        };
        aicore_trellis_preview_fn previewLambda =
                [](void* user, int stage, int step, int total, const void* data,
                   int len) {
                    auto* self = static_cast<TrellisWorker*>(user);
                    self->emitPreviewBlob(stage, step, total, data, len);
                };
        aicore_trellis_preview_fn previewFn =
                m_settings.livePreview ? previewLambda : nullptr;
        mesh = aicore_trellis_generate_ex(
                ctx, imageBytes.constData(), imageBytes.size(), &params,
                progressFn, this, previewFn, this, err, sizeof(err));
    }
    const double elapsedMs = static_cast<double>(timer.elapsed());
    // Close the trailing stage's wall-time bucket.
    if (m_lastStage >= 0) {
        m_stageMs[m_lastStage] += static_cast<double>(m_stageTimer.elapsed());
    }

    if (!mesh) {
        const QString failure = QString::fromUtf8(err);
        emit logMessage(
                QStringLiteral("[TRELLIS] Generation failed: %1").arg(failure));
        if (failure.contains(QStringLiteral("OutOfDeviceMemory")) ||
            failure.contains(QStringLiteral("out of memory"),
                             Qt::CaseInsensitive)) {
            emit logMessage(QStringLiteral(
                    "[TRELLIS] The GPU ran out of memory mid-run. Switch "
                    "Device to cpu (or cuda on an NVIDIA card), pick the "
                    "Coarse preset or the q8 quantization chain, close other "
                    "GPU applications, and retry."));
        }
        aicore_trellis_free(ctx);
        m_ctx = nullptr;
        return false;
    }

    TrellisRunResult result;
    result.sourceImage = m_settings.inputPath;
    result.presetName = m_settings.presetName;
    result.totalRuntimeMs = elapsedMs;
    result.backend = QString::fromUtf8(aicore_trellis_backend(ctx));
    result.quantization = m_settings.quantization;
    result.stageMs = m_stageMs;

    const int nv = aicore_trellis_mesh_n_verts(mesh);
    const int nt = aicore_trellis_mesh_n_tris(mesh);
    if (nv > 0) {
        const float* v = aicore_trellis_mesh_verts(mesh);
        const float* nn = aicore_trellis_mesh_normals(mesh);
        result.verts.resize(nv * 3);
        result.normals.resize(nv * 3);
        std::memcpy(result.verts.data(), v, sizeof(float) * nv * 3);
        if (nn) {
            std::memcpy(result.normals.data(), nn, sizeof(float) * nv * 3);
        }
    }
    if (nt > 0) {
        const int* t = aicore_trellis_mesh_tris(mesh);
        result.tris.resize(nt * 3);
        std::memcpy(result.tris.data(), t, sizeof(int) * nt * 3);
    }
    if (aicore_trellis_mesh_has_pbr(mesh)) {
        const float* pbr = aicore_trellis_mesh_pbr(mesh);
        result.pbr.resize(nv * 6);
        std::memcpy(result.pbr.data(), pbr, sizeof(float) * nv * 6);
        result.hasPbr = true;
    }
    // Decoded dual grid: carried through for standalone re-texturing
    // (aicore_trellis_texture_mesh with the sidecar grid path).
    result.gridRes = aicore_trellis_mesh_grid_res(mesh);
    const int nvox = aicore_trellis_mesh_grid_nvox(mesh);
    if (result.gridRes > 0 && nvox > 0) {
        const float* gf = aicore_trellis_mesh_grid_feats(mesh);
        const int* gc = aicore_trellis_mesh_grid_coords(mesh);
        result.gridFeats.resize(nvox * 7);
        result.gridCoords.resize(nvox * 3);
        std::memcpy(result.gridFeats.data(), gf, sizeof(float) * nvox * 7);
        std::memcpy(result.gridCoords.data(), gc, sizeof(int) * nvox * 3);
    }
    // AI background-removal result: wrap the borrowed RGBA buffer in a QImage
    // and detach with a deep copy, since the mesh (and its buffers) is freed
    // right below. QImage::Format_RGBA8888 matches the pipeline's byte order.
    if (aicore_trellis_mesh_has_rmbg(mesh)) {
        const uint8_t* rgba = aicore_trellis_mesh_rmbg_rgba(mesh);
        const int w = aicore_trellis_mesh_rmbg_w(mesh);
        const int h = aicore_trellis_mesh_rmbg_h(mesh);
        if (rgba && w > 0 && h > 0) {
            QImage img(rgba, w, h, w * 4, QImage::Format_RGBA8888);
            result.rmbgImage = img.copy();
        }
    }
    aicore_trellis_mesh_free(mesh);

    // Final strip preview: shaded vertex-splat render, O(nv) — see
    // renderResultPreview. Runs before the GLB bake so the strip thumbnail
    // is ready when the result lands.
    result.previewImage =
            renderResultPreview(result.verts, result.normals, result.pbr, 256);
    // Mesh chip gets a geometry-only render: assigning the textured image to
    // the Mesh/Texture/GLB chips alike made the three read as one broken
    // stage. Untextured runs already render amber here, so reuse it.
    result.geometryPreview =
            result.hasPbr ? renderResultPreview(result.verts, result.normals,
                                                QVector<float>(), 256)
                          : result.previewImage;

    // Bake the UV-atlas textured GLB here on the worker thread: the
    // add-to-DB path imports it for the full PBR material display (vertex
    // colours alone cannot express metallic/roughness, so the mesh would
    // render as flat diffuse base colour), and the GLB export reuses the
    // same bytes instead of re-baking on the GUI thread. Failure leaves the
    // result in vertex-colour fallback mode.
    double glbBakeMs = 0.0;
    if (m_settings.runMode == TrellisRunMode::GeometryOnly) {
        emit logMessage(QStringLiteral(
                "[TRELLIS] GLB bake skipped (geometry-only run) — use "
                "Generate + GLB or the Export page to bake the textured "
                "GLB."));
    } else if (result.hasPbr && !result.verts.isEmpty() &&
               !result.tris.isEmpty()) {
        emit logMessage(QStringLiteral(
                "[TRELLIS] Baking UV-atlas textured GLB (2048) on the worker "
                "thread (CPU-bound; complex meshes can take minutes, see the "
                "app log for per-stage timings)..."));
        QElapsedTimer glbTimer;
        glbTimer.start();
        char glbErr[512] = {0};
        int glbLen = 0;
        uint8_t* glb = aicore_trellis_bake_glb_ex(
                result.verts.constData(), result.verts.size() / 3,
                result.tris.constData(), result.tris.size() / 3,
                result.pbr.constData(), 2048, 0, &bakeProgressTrampoline, this,
                &m_bakeCancel, &glbLen, glbErr, sizeof(glbErr));
        if (!glb && std::strstr(glbErr, "cancelled")) {
            emit logMessage(QStringLiteral(
                    "[TRELLIS] GLB bake cancelled — the mesh stays "
                    "available for a re-bake."));
            aicore_trellis_free_buffer(glb);
            aicore_trellis_free(ctx);
            m_ctx = nullptr;
            return false;
        }
        if (glb && glbLen > 0) {
            result.glb = QByteArray(reinterpret_cast<const char*>(glb), glbLen);
            emit logMessage(
                    QStringLiteral("[TRELLIS] Textured GLB baked (%1 MB)")
                            .arg(glbLen / (1024.0 * 1024.0), 0, 'f', 1));
        } else {
            emit logMessage(
                    QStringLiteral("[TRELLIS] GLB bake failed (%1); falling "
                                   "back to vertex colours")
                            .arg(glbErr[0] ? QString::fromUtf8(glbErr)
                                           : QStringLiteral("unknown error")));
        }
        if (glb) {
            aicore_trellis_free_buffer(glb);
        }
        glbBakeMs = static_cast<double>(glbTimer.elapsed());
    }

    emit logMessage(QStringLiteral("[TRELLIS] Mesh %1 verts / %2 tris "
                                   "generated in %3 ms (backend %4, %5 chain)")
                            .arg(nv)
                            .arg(nt)
                            .arg(elapsedMs, 0, 'f', 0)
                            .arg(result.backend)
                            .arg(result.quantization));
    emitRunSummary(result, glbBakeMs);

    aicore_trellis_free(ctx);
    m_ctx = nullptr;

    emit resultReady(result);
    return true;
}

// Post-run console digest: end-to-end wall time, per-stage wall times with
// one-line explanations, and mesh/quality stats. One logMessage per line so
// the dialog console renders them in order; the summary complements (does
// not replace) the one-line mesh log above.
void TrellisWorker::emitRunSummary(const TrellisRunResult& result,
                                   double glbBakeMs) {
    struct StageInfo {
        int stage;
        const char* name;
        const char* hint;
    };
    static const StageInfo kStages[] = {
            {AICORE_TRELLIS_STAGE_PREPROCESS, "preprocess",
             "image preprocess (+RMBG)"},
            {AICORE_TRELLIS_STAGE_DINO, "dino", "DINO conditioning encode"},
            {AICORE_TRELLIS_STAGE_SS_FLOW, "ss_flow",
             "sparse structure flow sampling"},
            {AICORE_TRELLIS_STAGE_SS_DEC, "ss_dec",
             "occupancy decode (CPU conv3d)"},
            {AICORE_TRELLIS_STAGE_SLAT_FLOW, "slat_flow",
             "shape-SLAT flow sampling"},
            {AICORE_TRELLIS_STAGE_UPSAMPLE, "upsample",
             "cascade scaffold upsample"},
            {AICORE_TRELLIS_STAGE_SLAT_FLOW_HR, "slat_flow_hr",
             "HR shape-SLAT flow sampling"},
            {AICORE_TRELLIS_STAGE_SHAPE_DEC, "shape_dec",
             "mesh decode (512^3 grid)"},
            {AICORE_TRELLIS_STAGE_SHAPE_DEC_HR, "shape_dec_hr",
             "mesh decode (1024^3 grid)"},
            {AICORE_TRELLIS_STAGE_MESH, "mesh",
             "marching cubes + cleanup (CPU)"},
            {AICORE_TRELLIS_STAGE_TEXTURE, "texture",
             "PBR texture flow + decode"},
    };

    // 1000-grouped integer ("1564279" -> "1,564,279") for the mesh counters.
    auto grouped = [](int v) {
        QString s = QString::number(v);
        for (int pos = s.size() - 3; pos > 0; pos -= 3) s.insert(pos, u',');
        return s;
    };
    // Fixed-width row tags keep the digest block scannable: "Preset : ",
    // "Device : ", ... (8-column left-justified label + colon).
    auto tag = [](const char* text) {
        return QStringLiteral("[TRELLIS] %1: ")
                .arg(QString::fromLatin1(text).leftJustified(7));
    };

    emit logMessage(QStringLiteral(
            "[TRELLIS] ---- Run summary ------------------------------------"
            "-------"));
    emit logMessage(tag("Preset") +
                    QStringLiteral("%1 (%2)")
                            .arg(result.presetName.isEmpty()
                                         ? QStringLiteral("(custom)")
                                         : result.presetName)
                            .arg(result.quantization));
    emit logMessage(tag("Device") +
                    QStringLiteral("%1 | Seed: %2")
                            .arg(result.backend,
                                 m_settings.seed == 0
                                         ? QStringLiteral("random")
                                         : QString::number(m_settings.seed)));
    emit logMessage(
            tag("Grid") +
            (result.gridRes > 0
                     ? QStringLiteral("%1^3 dual grid").arg(result.gridRes)
                     : QStringLiteral("coarse 64^3 occupancy")));
    emit logMessage(
            tag("Time") +
            (glbBakeMs > 0.0
                     ? QStringLiteral("end-to-end %1 s (+ GLB bake %2 s)")
                               .arg(result.totalRuntimeMs / 1000.0, 0, 'f', 1)
                               .arg(glbBakeMs / 1000.0, 0, 'f', 1)
                     : QStringLiteral("end-to-end %1 s")
                               .arg(result.totalRuntimeMs / 1000.0, 0, 'f',
                                    1)));
    emit logMessage(tag("Stages") +
                    QStringLiteral("(time = share of the end-to-end "
                                   "inference):"));
    emit logMessage(
            QStringLiteral("[TRELLIS]     %1 %2 %3   %4")
                    .arg(QString::fromLatin1("stage").leftJustified(12),
                         QString::fromLatin1("what it does").leftJustified(30),
                         QString::fromLatin1("time").rightJustified(8),
                         QString::fromLatin1("share").rightJustified(6)));
    for (const StageInfo& s : kStages) {
        const auto it = result.stageMs.constFind(s.stage);
        if (it == result.stageMs.constEnd() || result.totalRuntimeMs <= 0.0) {
            continue;
        }
        const double sec = *it / 1000.0;
        const double pct = *it / result.totalRuntimeMs * 100.0;
        emit logMessage(
                QStringLiteral("[TRELLIS]     %1 %2 %3 s %4%")
                        .arg(QString::fromLatin1(s.name).leftJustified(12),
                             QString::fromLatin1(s.hint).leftJustified(30),
                             QString::number(sec, 'f', 2).rightJustified(8),
                             QString::number(pct, 'f', 1).rightJustified(6)));
    }
    emit logMessage(tag("Mesh") +
                    QStringLiteral("%1 verts / %2 tris")
                            .arg(grouped(result.verts.size() / 3),
                                 grouped(result.tris.size() / 3)));
    emit logMessage(
            tag("Quality") +
            QStringLiteral("%1 | GLB %2 | geometry sha %3")
                    .arg(result.hasPbr ? QStringLiteral("PBR textured")
                                       : QStringLiteral("untextured"),
                         result.glb.isEmpty()
                                 ? QStringLiteral("none")
                                 : QStringLiteral("%1 MB").arg(
                                           result.glb.size() /
                                                   (1024.0 * 1024.0),
                                           0, 'f', 1),
                         QString::fromLatin1(
                                 QCryptographicHash::hash(
                                         QByteArray::fromRawData(
                                                 reinterpret_cast<const char*>(
                                                         result.verts
                                                                 .constData()),
                                                 static_cast<qsizetype>(
                                                         result.verts.size()) *
                                                         static_cast<qsizetype>(
                                                                 sizeof(float))),
                                         QCryptographicHash::Sha256)
                                         .toHex()
                                         .left(12))));
}

// BakeOnly: re-bake a retained result (mesh arrays only — no inference
// context, no GPU, no input image) with live stage progress in the console
// and cooperative cancellation.
bool TrellisWorker::runBakeOnly() {
    const TrellisRunResult& in = m_settings.bakeInput;
    if (in.verts.isEmpty() || in.tris.isEmpty()) {
        emit logMessage(
                QStringLiteral("[TRELLIS] Nothing to bake — run a "
                               "generation first."));
        return false;
    }
    emit logMessage(QStringLiteral("[TRELLIS] Baking GLB (%1 px) from the "
                                   "last generation...")
                            .arg(m_settings.bakeTextureSize));
    char err[512] = {0};
    int outLen = 0;
    m_bakeCancel = 0;
    uint8_t* bytes = aicore_trellis_bake_glb_ex(
            in.verts.constData(), in.verts.size() / 3, in.tris.constData(),
            in.tris.size() / 3, in.hasPbr ? in.pbr.constData() : nullptr,
            m_settings.bakeTextureSize, m_settings.bakeComponentFilter,
            &bakeProgressTrampoline, this, &m_bakeCancel, &outLen, err,
            sizeof(err));
    if (!bytes) {
        const QString e = QString::fromUtf8(err);
        if (e.contains(QStringLiteral("cancelled"))) {
            emit logMessage(
                    QStringLiteral("[TRELLIS] GLB bake cancelled — "
                                   "the retained result is "
                                   "unchanged; re-bake anytime."));
            emit taskFinished(false);
            return false;
        }
        emit logMessage(QStringLiteral("[TRELLIS] GLB bake failed: %1").arg(e));
        emit taskFinished(false);
        return false;
    }
    emit logMessage(QStringLiteral("[TRELLIS] GLB baked (%1 MB)")
                            .arg(outLen / (1024.0 * 1024.0), 0, 'f', 1));
    emit bakeGlbReady(QByteArray(reinterpret_cast<const char*>(bytes), outLen));
    aicore_trellis_free_buffer(bytes);
    emit taskFinished(true);
    return true;
}

void TrellisWorker::bakeProgressTrampoline(const char* stage,
                                           double elapsed_s,
                                           void* user) {
    static_cast<TrellisWorker*>(user)->onBakeProgress(QString::fromUtf8(stage),
                                                      elapsed_s);
}

void TrellisWorker::onBakeProgress(const QString& stage, double elapsedS) {
    emit logMessage(QStringLiteral("[TRELLIS] bake: %1 (%2 s)")
                            .arg(stage)
                            .arg(elapsedS, 0, 'f', 1));
    // Drive the Export-page bake progress (busy bar + stage text) as well;
    // the log line alone is invisible outside the Generate page.
    emit bakeProgress(stage, elapsedS);
}

#endif  // AICore_ENABLED

// ── TrellisPrintWorker: CGAL Alpha-Wrap print remesh of the last result ────

TrellisPrintWorker::TrellisPrintWorker(const TrellisPrintRequest& request,
                                       QObject* parent)
    : QThread(parent), m_request(request) {}

TrellisPrintWorker::~TrellisPrintWorker() {
    // Same join-only discipline as TrellisWorker: the wrap state belongs to
    // the worker thread; this destructor never touches it, it just joins a
    // still-running thread so nothing leaks behind a fast shutdown.
    if (isRunning()) {
        requestInterruption();
        wait(5000);
    }
}

void TrellisPrintWorker::run() {
    // QThread entry: an exception escaping here terminates the whole
    // process. The AICore C ABI fences its own exceptions; this catch-all
    // is the last-resort guard for this worker thread.
#ifdef AICore_ENABLED
    try {
        // Alpha-Wrap ratios as fractions of the component-filtered input
        // bounding-box diagonal (trellis_capi.h recommended start).
        constexpr float kAlphaRatio = 0.01f;
        constexpr float kOffsetRatio = 0.01f / 30.0f;

        TrellisPrintResult out;
        out.sourceNVerts = m_request.source.verts.size() / 3;
        out.sourceNTris = m_request.source.tris.size() / 3;
        if (out.sourceNVerts <= 0 || out.sourceNTris <= 0) {
            emit logMessage(QStringLiteral(
                    "[TRELLIS] Print wrap skipped: no generation result."));
            emit taskFinished(false);
            return;
        }

        QElapsedTimer timer;
        timer.start();
        char err[512] = {0};
        aicore_trellis_mesh* wrap = aicore_trellis_prepare_print_mesh(
                m_request.source.verts.constData(), out.sourceNVerts,
                m_request.source.tris.constData(), out.sourceNTris,
                m_request.source.hasPbr ? m_request.source.pbr.constData()
                                        : nullptr,
                m_request.componentFilter, kAlphaRatio, kOffsetRatio, err,
                sizeof(err));
        if (!wrap) {
            emit logMessage(
                    QStringLiteral("[TRELLIS] Print wrap failed: %1")
                            .arg(err[0] ? QString::fromUtf8(err)
                                        : QStringLiteral("unknown error")));
            emit taskFinished(false);
            return;
        }
        out.wrapMs = timer.elapsed();

        // Copy the wrap buffers out before the handle is freed.
        const int nv = aicore_trellis_mesh_n_verts(wrap);
        const int nt = aicore_trellis_mesh_n_tris(wrap);
        if (nv > 0) {
            out.verts.resize(nv * 3);
            std::memcpy(out.verts.data(), aicore_trellis_mesh_verts(wrap),
                        sizeof(float) * nv * 3);
            if (const float* nn = aicore_trellis_mesh_normals(wrap)) {
                out.normals.resize(nv * 3);
                std::memcpy(out.normals.data(), nn, sizeof(float) * nv * 3);
            }
        }
        if (nt > 0) {
            out.tris.resize(nt * 3);
            std::memcpy(out.tris.data(), aicore_trellis_mesh_tris(wrap),
                        sizeof(int) * nt * 3);
        }
        if (aicore_trellis_mesh_has_pbr(wrap)) {
            out.pbr.resize(nv * 6);
            std::memcpy(out.pbr.data(), aicore_trellis_mesh_pbr(wrap),
                        sizeof(float) * nv * 6);
            out.hasPbr = true;
        }
        aicore_trellis_mesh_free(wrap);

        emit logMessage(
                QStringLiteral("[TRELLIS] Alpha-Wrap print mesh ready: %1 "
                               "verts / %2 tris (watertight) in %3 ms")
                        .arg(nv)
                        .arg(nt)
                        .arg(out.wrapMs, 0, 'f', 0));

        // Projected PBR GLB: wrap geometry as target, dense textured source
        // for the per-texel projection. The C API requires source PBR, so
        // untextured results stay on the vertex-colour path.
        if (m_request.bakeGlb && out.hasPbr) {
            emit logMessage(QStringLiteral(
                    "[TRELLIS] Baking projected PBR GLB of the print mesh "
                    "(closest-surface projection from the dense source)..."));
            timer.restart();
            int glbLen = 0;
            uint8_t* glb = aicore_trellis_bake_projected_glb(
                    out.verts.constData(), nv, out.tris.constData(), nt,
                    m_request.source.verts.constData(), out.sourceNVerts,
                    m_request.source.tris.constData(), out.sourceNTris,
                    m_request.source.pbr.constData(), m_request.textureSize,
                    m_request.componentFilter, &glbLen, err, sizeof(err));
            out.bakeMs = timer.elapsed();
            if (glb && glbLen > 0) {
                out.glb =
                        QByteArray(reinterpret_cast<const char*>(glb), glbLen);
                emit logMessage(
                        QStringLiteral("[TRELLIS] Projected print GLB baked "
                                       "(%1 MB) in %2 ms")
                                .arg(glbLen / (1024.0 * 1024.0), 0, 'f', 1)
                                .arg(out.bakeMs, 0, 'f', 0));
            } else {
                emit logMessage(
                        QStringLiteral("[TRELLIS] Projected GLB bake failed "
                                       "(%1); keeping the vertex-colour "
                                       "print mesh")
                                .arg(err[0] ? QString::fromUtf8(err)
                                            : QStringLiteral("unknown error")));
            }
            if (glb) {
                aicore_trellis_free_buffer(glb);
            }
        }

        emit printResultReady(out);
        emit taskFinished(true);
    } catch (const std::exception& e) {
        emit logMessage(QStringLiteral("[TRELLIS] Unexpected print-wrap "
                                       "failure: %1")
                                .arg(QString::fromUtf8(e.what())));
        emit taskFinished(false);
    }
#else
    Q_UNUSED(m_request);
    emit logMessage(QStringLiteral("[TRELLIS] AICore not enabled."));
    emit taskFinished(false);
#endif
}
