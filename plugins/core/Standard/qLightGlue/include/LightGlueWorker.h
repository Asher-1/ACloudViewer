#pragma once

#include <QThread>
#include <QPointF>
#include <QVector>

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
        double minScore = 0.1;
        int matcherType = 2;
        int maxKeypoints = 2048;
        int maxResize = 1024;
    };

    explicit LightGlueWorker(const Settings& settings, QObject* parent = nullptr);

    void releaseContextOnMainThread();

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
    struct aicore_lightglue_ctx* m_pendingCtx = nullptr;
};

Q_DECLARE_METATYPE(LightGlueRunResult)
