// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QObject>
#include <QString>

#include "CVPluginAPI.h"

class QNetworkAccessManager;
class QNetworkReply;
class QFile;

/** Shared HTTPS model downloader for GGUF / large asset plugins. */
class CVPLUGIN_LIB_API ecvModelDownloader : public QObject {
    Q_OBJECT

public:
    struct Request {
        QString url;
        QString destPath;
        /** Reject cached files smaller than this (bytes). Default 1 MiB. */
        qint64 minValidBytes = 1024 * 1024;
    };

    explicit ecvModelDownloader(QObject* parent = nullptr);
    ~ecvModelDownloader() override;

    static bool isValidCachedFile(const QString& path, qint64 minBytes = 1024 * 1024);
    static void removeInvalidCacheFile(const QString& path, qint64 minBytes = 1024 * 1024);

    /** Human-readable size (B / KB / MB / GB). */
    static QString formatFileSize(qint64 bytes);
    /** e.g. "12.3 MB / 45.6 MB (27%)". */
    static QString formatDownloadProgress(qint64 received, qint64 total);

    bool isBusy() const { return m_busy; }

public slots:
    void download(const Request& request);
    void cancel();

signals:
    void progress(qint64 received, qint64 total);
    void logMessage(const QString& message);
    /** ok=true when destPath contains a valid file (size >= minValidBytes). */
    void finished(bool ok, const QString& destPath);

private:
    void cleanupActiveReply();

    QNetworkAccessManager* m_net = nullptr;
    QNetworkReply* m_reply = nullptr;
    QFile* m_outFile = nullptr;
    QString m_tmpPath;
    QString m_destPath;
    qint64 m_minValidBytes = 0;
    bool m_busy = false;
};
