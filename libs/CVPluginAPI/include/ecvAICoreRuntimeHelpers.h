// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Qt-side runtime helpers shared by AICore-based plugin workers (qDA3,
// qDeepLSD, qFaceDetect, qGKD, qLightGlue, qLingbotMap, qRFDetr, qRMBG,
// qSAM3, qTrellis, qYOLO).
//
// Qt-flavored on purpose: the Qt-free RAII layer lives in
// aicore/runtime_raii.h (AICore public headers); this header only adds
// what needs QImage memory-layout knowledge or the plugin worker thread
// affinity convention. Header-only: no out-of-line symbols, nothing to
// link.
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QImage>
#include <QString>
#include <QtGlobal>

#include "aicore/image_view.h"
#include "aicore/runtime_capi.h"
#include "aicore/runtime_raii.h"

namespace ecvAICoreRuntime {

// ---------------------------------------------------------------------------
//  Device task lock (Qt convenience overload)
// ---------------------------------------------------------------------------

/** Serializes this worker against every other AICore inference task on the
 *  same device (live video loops, other plugin workers): a backend's state
 *  machine is not safe under concurrent graph compute from two threads, and
 *  the shared device queue lock is the process-wide mutex that keeps
 *  command buffers from racing (a failed command buffer poisons the backend
 *  for the rest of the process). Qt convenience overload of aicore::
 *  runtime::DeviceTaskLock(const char*) — worker settings carry QString
 *  devices. Check isLocked() on the returned guard before running
 *  inference; never nest the guards on one thread. */
inline aicore::runtime::DeviceTaskLock makeDeviceTaskLock(
        const QString& device) {
    const QByteArray bytes = device.toUtf8();
    return aicore::runtime::DeviceTaskLock(bytes.constData());
}

// ---------------------------------------------------------------------------
//  QImage -> aicore_image_view
// ---------------------------------------------------------------------------

/** Maps a QImage onto a borrowed aicore_image_view (no copy, no ownership
 *  transfer). The QImage must outlive every use of the returned view.
 *
 *  Directly supported formats (little-endian memory order assumed):
 *    Format_RGB888      -> AICORE_IMAGE_RGB8
 *    Format_BGR888      -> AICORE_IMAGE_BGR8 (Qt >= 5.14; on older Qt5 no
 *                          QImage can carry this format, so the case is
 *                          compiled out and other formats map unchanged)
 *    Format_RGBA8888    -> AICORE_IMAGE_RGBA8
 *    Format_ARGB32      -> AICORE_IMAGE_BGRA8 (0xAARRGGBB is stored as
 *                          B,G,R,A bytes on little-endian)
 *    Format_Grayscale8  -> AICORE_IMAGE_GRAY8
 *
 *  On any other format the returned view has data == nullptr; convert
 *  once with QImage::convertToFormat(), keep the converted QImage alive
 *  for the duration of the inference call, and call again.
 *  row_stride_bytes is the real bytesPerLine(); never assume tightly
 *  packed rows (see the AICore image-view contract, skill §4).
 */
inline aicore_image_view makeImageView(const QImage& image) {
    aicore_image_view view{};
    if (image.isNull()) {
        return view;
    }
    view.width = image.width();
    view.height = image.height();
    view.row_stride_bytes = static_cast<size_t>(image.bytesPerLine());
    view.data = reinterpret_cast<const uint8_t*>(image.constBits());
    switch (image.format()) {
        case QImage::Format_RGB888:
            view.format = AICORE_IMAGE_RGB8;
            break;
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
        case QImage::Format_BGR888:
            view.format = AICORE_IMAGE_BGR8;
            break;
#endif
        case QImage::Format_RGBA8888:
            view.format = AICORE_IMAGE_RGBA8;
            break;
        case QImage::Format_Grayscale8:
            view.format = AICORE_IMAGE_GRAY8;
            break;
        case QImage::Format_ARGB32:
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
            view.format = AICORE_IMAGE_BGRA8;
#else
            view.data = nullptr;  // big-endian ARGB32 memory is RGBA; bail out
#endif
            break;
        default:
            view.data = nullptr;  // unsupported format: caller must convert
            break;
    }
    return view;
}

// ---------------------------------------------------------------------------
//  Pending task-context holder (worker produces, main thread releases)
// ---------------------------------------------------------------------------

/** Holds a task context created on a worker thread until the owner
 *  (dialog / main thread) destroys this holder or calls release().
 *
 *  Rationale: aicore_*_free() tears down GPU state; freeing from the
 *  worker while the render thread is still drawing races in practice, so
 *  every AICore plugin worker stashes the context and lets the main
 *  thread release it (the historical releaseContextOnMainThread pattern
 *  in qDA3/qYOLO/qGKD/…, now shared).
 *
 *  \p Ctx is an opaque task context type (e.g. aicore_depth_ctx) and
 *  \p FreeFn the matching free function (e.g. aicore_depth_free).
 */
template <typename Ctx>
class PendingContext {
public:
    using FreeFn = void (*)(Ctx*);

    PendingContext() = default;
    ~PendingContext() { release(); }

    PendingContext(const PendingContext&) = delete;
    PendingContext& operator=(const PendingContext&) = delete;

    /** Releases any previously stashed context, then stores \p ctx. */
    void stash(Ctx* ctx, FreeFn free_fn) {
        release();
        ctx_ = ctx;
        free_fn_ = free_fn;
    }

    /** Detaches without freeing (ownership moves back to the caller). */
    Ctx* take() {
        Ctx* ctx = ctx_;
        ctx_ = nullptr;
        free_fn_ = nullptr;
        return ctx;
    }

    /** Frees the stashed context with its matching free function. */
    void release() {
        if (ctx_ && free_fn_) {
            free_fn_(ctx_);
        }
        ctx_ = nullptr;
        free_fn_ = nullptr;
    }

    explicit operator bool() const { return ctx_ != nullptr; }

private:
    Ctx* ctx_ = nullptr;
    FreeFn free_fn_ = nullptr;
};

// ---------------------------------------------------------------------------
//  One-shot release for a pending raw context pointer
// ---------------------------------------------------------------------------

/** Releases a pending raw context pointer with its matching free function
 *  and nulls it — the minimal shared form of the historical
 *  `releaseContextOnMainThread()` bodies (the worker creates the context on
 *  the inference thread; the dialog/main thread frees it so GPU teardown
 *  never races the render thread).
 *
 *  Use this when the worker keeps using the raw pointer during inference
 *  (the common case: the pending slot is only the ownership handoff).
 *  Use PendingContext<Ctx> instead when a slot must own-and-guard a
 *  context between handoffs (stash/take/release lifecycle).
 */
template <typename Ctx>
inline void releasePending(Ctx*& ctx, void (*free_fn)(Ctx*)) {
    if (ctx) {
        free_fn(ctx);
        ctx = nullptr;
    }
}

}  // namespace ecvAICoreRuntime
