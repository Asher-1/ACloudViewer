// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "aicore/aliked_capi.h"
#include "aicore/backend_capi.h"
#include "aicore/deeplsd_capi.h"
#include "aicore/depth_capi.h"
#include "aicore/facedetect_capi.h"
#include "aicore/gaussian_capi.h"
#include "aicore/lightglue_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/runtime_capi.h"
#include "aicore/runtime_raii.h"
#include "aicore/yolo_capi.h"

// The legacy process-wide cancel / inference-lock entry points are marked
// AICORE_LEGACY_API (deprecated). This contract test intentionally verifies
// they still work, so suppress the deprecation warnings for the whole file.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

namespace {
using Token = std::unique_ptr<aicore_cancel_token,
                              decltype(&aicore_cancel_token_free)>;

TEST(RuntimeContract, ShutdownIsIdempotentAndNullImagesFailSynchronously) {
    aicore_runtime_shutdown();
    aicore_runtime_shutdown();
    aicore_aliked_shutdown();
    aicore_deeplsd_shutdown();
    aicore_depth_shutdown();
    aicore_gaussian_shutdown();
    aicore_lightglue_shutdown();
    aicore_rfdetr_shutdown();
    aicore_yolo_shutdown();
    EXPECT_NE(aicore_depth_depth_image(nullptr, nullptr, nullptr), 0);
    EXPECT_NE(aicore_facedetect_detect_rgb(nullptr, nullptr, 0, 0), 0);
    EXPECT_NE(aicore_rfdetr_detect_rgb(nullptr, nullptr, 0, 0, 0.0F, 0), 0);
    EXPECT_NE(aicore_yolo_detect_rgb(nullptr, nullptr, 0, 0), 0);
}

TEST(RuntimeContract, NestedCancellationRestoresOuterScope) {
    Token outer(aicore_cancel_token_new(), aicore_cancel_token_free);
    Token inner(aicore_cancel_token_new(), aicore_cancel_token_free);
    ASSERT_NE(outer, nullptr);
    ASSERT_NE(inner, nullptr);
    aicore_cancel_token_request(outer.get());
    aicore::runtime::CancelScope outer_scope(outer.get());
    EXPECT_TRUE(aicore_cancel_requested());
    aicore_cancel_token_reset(outer.get());
    EXPECT_FALSE(aicore_cancel_requested());
    {
        aicore::runtime::CancelScope inner_scope(inner.get());
        aicore_cancel_token_request(inner.get());
        EXPECT_TRUE(aicore_cancel_requested());
    }
    EXPECT_FALSE(aicore_cancel_requested());
}

TEST(RuntimeContract, LegacyCancellationResetsAtScopeBoundaries) {
    aicore_cancel_begin();
    EXPECT_FALSE(aicore_cancel_requested());
    aicore_cancel_request();
    EXPECT_TRUE(aicore_cancel_requested());
    aicore_cancel_end();
    EXPECT_FALSE(aicore_cancel_requested());
}

TEST(RuntimeContract, DeviceQueueRejectsNestedAcquisitionAndReleases) {
    {
        aicore::runtime::DeviceTaskLock lock("cpu");
        ASSERT_TRUE(lock.isLocked());
        EXPECT_NE(aicore_device_task_try_lock("cpu"), 0);
    }
    const int status = aicore_device_task_try_lock("cpu");
    EXPECT_EQ(status, 0);
    if (status == 0) aicore_device_task_unlock();
}

TEST(RuntimeContract, QueuedTaskObservesCancellationBeforeAcquiringDevice) {
    Token queued(aicore_cancel_token_new(), aicore_cancel_token_free);
    ASSERT_NE(queued, nullptr);
    aicore::runtime::DeviceTaskLock lock("cpu");
    ASSERT_TRUE(lock.isLocked());
    std::atomic<int> result{-2};
    std::thread waiter([&]() {
        const int status =
                aicore_device_task_lock_cancelable("cpu", queued.get());
        result.store(status);
        if (status == 0) aicore_device_task_unlock();
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    aicore_cancel_token_request(queued.get());
    waiter.join();
    EXPECT_EQ(result.load(), 1);
}

TEST(RuntimeContract, LegacyInferenceLockRejectsNestedAcquisitionAndReleases) {
    ASSERT_EQ(aicore_inference_lock(), 0);
    EXPECT_EQ(aicore_inference_try_lock(), -1);
    aicore_inference_unlock();
    const int status = aicore_inference_try_lock();
    EXPECT_EQ(status, 0);
    if (status == 0) aicore_inference_unlock();
}

TEST(RuntimeContract, CpuGpuAndAutomaticCapabilityMasksResolve) {
    EXPECT_NE(aicore_device_capabilities("cpu"), 0u);
    EXPECT_NE(aicore_device_capabilities("gpu"), 0u);
    EXPECT_NE(aicore_device_capabilities(nullptr), 0u);
}
}  // namespace

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
