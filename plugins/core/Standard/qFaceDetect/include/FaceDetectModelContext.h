// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>

struct aicore_facedetect_ctx;
struct aicore_cancel_token;

/** Binds a synchronous caller to one resolved device queue and cancel scope.
 */
class FaceDetectInferenceGuard {
public:
    explicit FaceDetectInferenceGuard(
            const QString& device = QStringLiteral("auto"));
    ~FaceDetectInferenceGuard();
    FaceDetectInferenceGuard(const FaceDetectInferenceGuard&) = delete;
    FaceDetectInferenceGuard& operator=(const FaceDetectInferenceGuard&) =
            delete;

private:
    aicore_cancel_token* m_cancelToken = nullptr;
    bool m_active = false;
};

#ifdef AICore_ENABLED

/** RAII wrapper for a loaded facedetect GGUF context (reuse across embed/auth).
 */
class FaceDetectModelContext {
public:
    FaceDetectModelContext() = default;
    ~FaceDetectModelContext();
    FaceDetectModelContext(const FaceDetectModelContext&) = delete;
    FaceDetectModelContext& operator=(const FaceDetectModelContext&) = delete;
    FaceDetectModelContext(FaceDetectModelContext&& other) noexcept;
    FaceDetectModelContext& operator=(FaceDetectModelContext&& other) noexcept;

    bool load(const QString& modelPath, const QString& device, int threads);
    /** Reload only when model path, device, or thread count changes. */
    bool ensureLoaded(const QString& modelPath,
                      const QString& device,
                      int threads);
    void release();

    aicore_facedetect_ctx* get() const { return m_ctx; }
    explicit operator bool() const { return m_ctx != nullptr; }

private:
    aicore_facedetect_ctx* m_ctx = nullptr;
    QString m_modelPath;
    QString m_device;
    int m_threads = 0;
};

#endif
