// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QPointF>
#include <QThread>
#include <QVector>

struct aicore_cancel_token;
struct aicore_lightglue_ctx;
namespace lightglue_plugin {
struct OwnedFeatures;
}

struct LightGlueMatchResult {
    int idx1 = -1;
    int idx2 = -1;
    float score = 0.f;
};

struct LightGlueRunResult {
    QVector<LightGlueMatchResult> matches;
    QVector<QPointF> keypoints0;
    QVector<QPointF> keypoints1;
    int nKeypoints0 = 0;
    int nKeypoints1 = 0;
    int imageWidth0 = 0;
    int imageHeight0 = 0;
    int imageWidth1 = 0;
    int imageHeight1 = 0;
    QString imagePath0;
    QString imagePath1;
    QString imageName0;
    QString imageName1;
    QString sourceName;
    QString resolvedDevice;
    double runtimeMs = 0.0;
};

class LightGlueWorker : public QThread {
    Q_OBJECT

public:
    enum class Mode { Match, ModelInfo };

    struct Settings {
        Mode mode = Mode::Match;
        QString modelPath;
        QStringList inputPaths;
        int threads = 0;
        QString device = "auto";
        double minScore = 0.5;
        int matcherType = 2;
        int maxKeypoints = 2048;
        int maxResize = 640;
    };

    explicit LightGlueWorker(const Settings& settings,
                             QObject* parent = nullptr);
    ~LightGlueWorker() override;

    void releaseContextOnMainThread();
    void requestTaskCancel();

#ifdef AICore_ENABLED
    // Per-step bodies shared by run() and the -LIGHTGLUE_MATCH command.

    /** Matcher context on \p device (settings.minScore / matcherType /
     *  threads). May return a context together with \p error; the caller
     *  frees any non-null return with aicore_lightglue_free. */
    static aicore_lightglue_ctx* loadContext(const Settings& settings,
                                             const QString& device,
                                             QString* error);
    /** Canonical extractor device ("cuda", "vulkan", "cpu"...) of a loaded
     *  matcher; \p fallback when it cannot be mapped. */
    static QString canonicalDevice(aicore_lightglue_ctx* ctx,
                                   const QString& fallback);
    /** RootSIFT (matcherType 1) or ALIKED (same-quant extractor next to the
     *  matcher GGUF) features of one image on settings.device. */
    static bool extractFeatures(const Settings& settings,
                                const QString& imagePath,
                                lightglue_plugin::OwnedFeatures* out,
                                QString* error);
    /** Matches f0 x f1 and fills \p out (settings.inputPaths name the pair).
     *  A GPU run returning 0 matches is retried on a lazily created CPU
     *  matcher stored in \p cpuFallback (caller frees it). */
    static bool matchFeatures(aicore_lightglue_ctx* ctx,
                              aicore_lightglue_ctx** cpuFallback,
                              const Settings& settings,
                              const lightglue_plugin::OwnedFeatures& f0,
                              const lightglue_plugin::OwnedFeatures& f1,
                              LightGlueRunResult* out,
                              QString* log,
                              QString* error);
#endif

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const LightGlueRunResult& result);
    void modelInfoReady(const QString& info);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runMatch();
    bool runModelInfo();
#endif

    Settings m_settings;
    aicore_cancel_token* m_cancelToken = nullptr;
    struct aicore_lightglue_ctx* m_pendingCtx = nullptr;
};

Q_DECLARE_METATYPE(LightGlueRunResult)
