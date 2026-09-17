// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Focused unit tests for the shared AICore runtime helper header
// (include/ecvAICoreRuntimeHelpers.h). Header-only under test; runs
// headless (pure QImage logic, no widgets, no QCoreApplication needed).

#include <QImage>
#include <cstdio>

#include "ecvAICoreRuntimeHelpers.h"

namespace {

int failures = 0;

void CHECK(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "runtime helpers: FAIL %s\n", what);
        ++failures;
    }
}

struct DummyCtx {
    int alive = 1;
};
int g_freed = 0;
void freeDummy(DummyCtx* ctx) {
    if (ctx) {
        --ctx->alive;
    }
    ++g_freed;
    delete ctx;
}

void testImageView() {
    const QImage rgb(5, 3, QImage::Format_RGB888);
    aicore_image_view view = ecvAICoreRuntime::makeImageView(rgb);
    CHECK(view.data != nullptr, "RGB888 maps to a borrowed view");
    CHECK(view.format == AICORE_IMAGE_RGB8, "RGB888 -> AICORE_IMAGE_RGB8");
    CHECK(view.width == 5 && view.height == 3, "dimensions pass through");
    CHECK(view.row_stride_bytes >= static_cast<size_t>(5 * 3),
          "stride carries the real bytesPerLine");

    const QImage gray(4, 2, QImage::Format_Grayscale8);
    CHECK(ecvAICoreRuntime::makeImageView(gray).format == AICORE_IMAGE_GRAY8,
          "Grayscale8 -> AICORE_IMAGE_GRAY8");

    const QImage argb(4, 2, QImage::Format_ARGB32);
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
    CHECK(ecvAICoreRuntime::makeImageView(argb).format == AICORE_IMAGE_BGRA8,
          "ARGB32 -> AICORE_IMAGE_BGRA8 on little-endian");
#endif

    // Deliberately unsupported format must yield an empty view so callers
    // convert once instead of feeding mislabeled memory to the engine.
    const QImage rgba(2, 2, QImage::Format_RGB16);
    CHECK(ecvAICoreRuntime::makeImageView(rgba).data == nullptr,
          "unsupported format -> data == nullptr");

    const QImage nullImage;
    CHECK(ecvAICoreRuntime::makeImageView(nullImage).data == nullptr,
          "null image -> data == nullptr");
}

void testPendingContext() {
    ecvAICoreRuntime::PendingContext<DummyCtx> pending;
    CHECK(!pending, "fresh holder is empty");

    pending.stash(new DummyCtx, &freeDummy);
    CHECK(static_cast<bool>(pending), "stash makes the holder non-empty");
    CHECK(g_freed == 0, "no free before release");

    DummyCtx* taken = pending.take();
    CHECK(taken != nullptr && !pending, "take detaches without freeing");
    CHECK(g_freed == 0, "take does not free");
    freeDummy(taken);

    pending.stash(new DummyCtx, &freeDummy);
    pending.release();
    CHECK(!pending && g_freed == 2, "release frees exactly once");

    pending.release();
    CHECK(g_freed == 2, "release is idempotent");
}

void testDeviceTaskLock() {
    // Unit-process scope: this binary holds no other device queue, so the
    // cpu queue must be acquirable and released exactly once (RAII).
    {
        auto lock = ecvAICoreRuntime::makeDeviceTaskLock(QStringLiteral("cpu"));
        CHECK(lock.isLocked(), "makeDeviceTaskLock(cpu) acquired the queue");
    }
    auto again = ecvAICoreRuntime::makeDeviceTaskLock(QStringLiteral("cpu"));
    CHECK(again.isLocked(),
          "queue is released again after the guard's scope ends");
}

}  // namespace

int main() {
    testImageView();
    testPendingContext();
    testDeviceTaskLock();
    if (failures != 0) {
        std::fprintf(stderr, "runtime helpers: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("runtime helpers: all checks passed\n");
    return 0;
}
