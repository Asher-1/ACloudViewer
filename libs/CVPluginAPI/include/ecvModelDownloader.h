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
        // Validation policy for the downloaded file. Both fields apply:
        //   1. the file must be at least minBytes long (floor for
        //      detecting truncated/empty responses), and
        //   2. when requireGgufMagic is true, the first four bytes must
        //      be the GGUF magic ("GGUF") — this is the canonical way to
        //      tell a real model file from an HTML error page or an
        //      empty/truncated download.
        // minBytes default is 64 KiB: large enough to reject empty/HTML
        // pages, small enough to admit the smallest ALIKED
        // (aliked-n16rot-q8_0.gguf is ~714 KiB on disk).
        qint64 minBytes = 64 * 1024;
        bool requireGgufMagic = true;
    };

    explicit ecvModelDownloader(QObject* parent = nullptr);
    ~ecvModelDownloader() override;

    /** Returns true if the file exists, meets minBytes, and (when
     *  requireGgufMagic is true) starts with the GGUF magic bytes. */
    static bool isValidCachedFile(const QString& path,
                                  qint64 minBytes = 64 * 1024,
                                  bool requireGgufMagic = true);
    static void removeInvalidCacheFile(const QString& path,
                                       qint64 minBytes = 64 * 1024,
                                       bool requireGgufMagic = true);

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
    bool m_requireGgufMagic = true;
    bool m_busy = false;
};
