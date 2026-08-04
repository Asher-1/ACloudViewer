// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FaceDetectModelContext.h"

#ifdef AICore_ENABLED
#include "aicore/facedetect_capi.h"
#include "aicore/runtime_capi.h"
#endif

FaceDetectInferenceGuard::FaceDetectInferenceGuard(const QString& device) {
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
    aicore_device_task_lock(device.toUtf8().constData());
    aicore_cancel_scope_begin(m_cancelToken);
    m_active = true;
#endif
}

FaceDetectInferenceGuard::~FaceDetectInferenceGuard() {
#ifdef AICore_ENABLED
    if (m_active) {
        aicore_cancel_scope_end(m_cancelToken);
        aicore_device_task_unlock();
    }
    aicore_cancel_token_free(m_cancelToken);
#endif
}

#ifdef AICore_ENABLED

FaceDetectModelContext::~FaceDetectModelContext() { release(); }

FaceDetectModelContext::FaceDetectModelContext(
        FaceDetectModelContext&& other) noexcept
    : m_ctx(other.m_ctx),
      m_modelPath(std::move(other.m_modelPath)),
      m_device(std::move(other.m_device)),
      m_threads(other.m_threads) {
    other.m_ctx = nullptr;
    other.m_threads = 0;
}

FaceDetectModelContext& FaceDetectModelContext::operator=(
        FaceDetectModelContext&& other) noexcept {
    if (this != &other) {
        release();
        m_ctx = other.m_ctx;
        m_modelPath = std::move(other.m_modelPath);
        m_device = std::move(other.m_device);
        m_threads = other.m_threads;
        other.m_ctx = nullptr;
        other.m_threads = 0;
    }
    return *this;
}

void FaceDetectModelContext::release() {
    if (m_ctx) {
        aicore_facedetect_free(m_ctx);
        m_ctx = nullptr;
    }
    m_modelPath.clear();
    m_device.clear();
    m_threads = 0;
}

bool FaceDetectModelContext::load(const QString& modelPath,
                                  const QString& device,
                                  int threads) {
    release();
    aicore_facedetect_options* opts = aicore_facedetect_options_new();
    if (!opts) return false;
    aicore_facedetect_options_set_device(opts, device.toUtf8().constData());
    aicore_facedetect_options_set_threads(opts, threads);
    m_ctx = aicore_facedetect_load_opts(modelPath.toUtf8().constData(), opts);
    aicore_facedetect_options_free(opts);
    if (!aicore_facedetect_is_ready(m_ctx)) {
        release();
        return false;
    }
    m_modelPath = modelPath;
    m_device = device;
    m_threads = threads;
    return true;
}

bool FaceDetectModelContext::ensureLoaded(const QString& modelPath,
                                          const QString& device,
                                          int threads) {
    if (m_ctx && m_modelPath == modelPath && m_device == device &&
        m_threads == threads) {
        return true;
    }
    return load(modelPath, device, threads);
}

#endif
