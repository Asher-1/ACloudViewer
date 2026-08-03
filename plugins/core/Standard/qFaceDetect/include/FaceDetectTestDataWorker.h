// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QThread>
#include <QString>
#include <QVector>

#include "FaceDetectTestData.h"
#include "FaceRegistryStore.h"

/** Background registration + multi-face verify for FriendsFaces test data. */
class FaceDetectTestDataWorker : public QThread {
    Q_OBJECT

public:
    struct Job {
        FaceDetectFriendsBundle bundle;
        QString registryPath;
        QString modelPath;
        QString device = QStringLiteral("auto");
        int threads = 0;
        float minDetectionScore = 0.5f;
        float authThreshold = 0.52f;
        bool extractZipFirst = false;
        QString zipPath;
        QString extractParentDir;
        bool registerGallery = true;
        bool runVerify = true;
        bool clearExistingEntries = true;
    };

    explicit FaceDetectTestDataWorker(QObject* parent = nullptr);
    void setJob(Job job);

signals:
    void phaseProgress(int current, int total, const QString& label);
    void logMessage(const QString& msg);
    void finished(bool ok, int registeredCount, int authFaceCount, int authMatchCount);

protected:
    void run() override;

private:
    Job m_job;
};
