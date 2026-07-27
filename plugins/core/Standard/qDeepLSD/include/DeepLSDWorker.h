// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QObject>
#include <QString>
#include <QThread>

struct DeepLSDRunResult {
    QString imagePath;
    QString imageName;
    int width = 0;
    int height = 0;
    QImage overlay;
    double runtimeMs = 0.0;
};

Q_DECLARE_METATYPE(DeepLSDRunResult)

class DeepLSDWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = "auto";
    };

    explicit DeepLSDWorker(const Settings& settings, QObject* parent = nullptr);
    void releaseContextOnMainThread();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const DeepLSDRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runExtract();
#endif

    Settings m_settings;
    struct aicore_deeplsd_ctx* m_pendingCtx = nullptr;
};
