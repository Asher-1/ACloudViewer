// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>
#include <QStringList>
#include <QThread>
#include <atomic>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

class SIBRViewerThread : public QThread {
    Q_OBJECT

public:
    enum class ViewerMode {
        ULR,
        ULRv2,
        TexturedMesh,
        PointBased,
        GaussianSplatting,
        RemoteGaussian,
        DatasetTool
    };

    SIBRViewerThread(ViewerMode mode,
                     const QString& datasetPath,
                     int width = 1280,
                     int height = 720,
                     QObject* parent = nullptr);

    SIBRViewerThread(const QString& toolName,
                     const QStringList& toolArgs,
                     QObject* parent = nullptr);

    ~SIBRViewerThread() override;

    void requestStop();
    bool isStopRequested() const;
    static bool hasActiveViewer();

    void setModelPath(const QString& path) { m_modelPath = path; }
    void setIteration(const QString& iter) { m_iteration = iter; }
    void setCudaDevice(int dev) { m_cudaDevice = dev; }
    void setNoInterop(bool val) { m_noInterop = val; }
    void setRemoteAddress(const QString& ip, int port) {
        m_remoteIP = ip;
        m_remotePort = port;
    }

signals:
    void viewerStarted(const QString& modeName);
    void viewerFinished(const QString& modeName, int exitCode);
    void viewerError(const QString& errorMsg);
    void viewerLog(const QString& message);

    // Emitted when the viewer session produces importable result files.
    // resultPath: absolute path to the file (PLY, OBJ, etc.)
    // description: human-readable label for the UI prompt
    void viewerResultReady(const QString& resultPath,
                           const QString& description);

protected:
    void run() override;

private:
    struct FakeArgs {
        std::vector<std::string> storage;
        std::vector<const char*> argv;
        int argc = 0;

        void build(
                const std::string& appName,
                const std::vector<std::pair<std::string, std::string>>& params);
    };

    int runULRViewer();
    int runULRv2Viewer();
    int runTexturedMeshViewer();
    int runPointBasedViewer();
    int runGaussianViewer();
    int runRemoteGaussianViewer();
    int runDatasetTool();

    static std::mutex s_sibrGlobalMutex;
    static std::atomic<int> s_activeViewerCount;

    ViewerMode m_mode;
    QString m_datasetPath;
    QString m_toolName;
    QStringList m_toolArgs;
    int m_width;
    int m_height;
    std::atomic<bool> m_stopRequested{false};

    QString m_modelPath;
    QString m_iteration;
    int m_cudaDevice = 0;
    bool m_noInterop = false;
    QString m_remoteIP = "127.0.0.1";
    int m_remotePort = 6009;
};
