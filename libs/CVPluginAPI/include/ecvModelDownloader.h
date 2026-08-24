// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QObject>
#include <QString>
#include <QUrl>

#include "CVPluginAPI.h"

class QNetworkAccessManager;
class QNetworkReply;
class QFile;
class QCryptographicHash;

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
        // When > 0, the downloaded file must be exactly this many bytes;
        // a mismatch deletes the file and fails the download. Used by
        // plugins whose assets are published with known sizes (e.g. HF LFS
        // mirrors) to catch truncated/CDN-error downloads that still carry
        // the GGUF magic.
        qint64 expectedSize = 0;
        // Hex-encoded SHA-256 (64 chars) of the expected file content, e.g.
        // the HF LFS oid. Computed WHILE downloading (streamed, no extra
        // I/O pass over the file); a mismatch deletes the file and fails
        // the download. This is the content-level integrity check —
        // expectedSize above only catches length changes.
        QByteArray expectedSha256;
    };

    explicit ecvModelDownloader(QObject* parent = nullptr);
    ~ecvModelDownloader() override;

    /** Returns true if the file exists, meets minBytes, matches expectedSize
     *  (when > 0), starts with the GGUF magic bytes (when requireGgufMagic),
     *  and matches expectedSha256 (when non-empty; computed by reading the
     *  whole file — use only for one-shot verification, not per-dialog
     *  presence checks on multi-GB models). */
    static bool isValidCachedFile(const QString& path,
                                  qint64 minBytes = 64 * 1024,
                                  bool requireGgufMagic = true,
                                  qint64 expectedSize = 0,
                                  const QByteArray& expectedSha256 = {});
    static void removeInvalidCacheFile(const QString& path,
                                       qint64 minBytes = 64 * 1024,
                                       bool requireGgufMagic = true,
                                       qint64 expectedSize = 0,
                                       const QByteArray& expectedSha256 = {});

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

public:
    // ── Hugging Face helpers (shared across plugins) ───────────────────

    /** Build the direct download URL for a file on a public Hugging Face
     *  model repo. Format:
     *    https://huggingface.co/<repoId>/resolve/main/<filename>
     *  This is the canonical URL that the HF Content Delivery Network
     *  (CDN) uses; ecvModelDownloader::download() will follow the 302
     *  → CDN redirect automatically, with no token required for public
     *  repos.
     *
     *  Plugins that publish their GGUF assets on HF can use this as a
     *  one-liner in place of hand-constructing the URL string. */
    static QUrl hfDownloadUrl(const QString& repoId, const QString& filename);

private:
    void cleanupActiveReply();

    QNetworkAccessManager* m_net = nullptr;
    QNetworkReply* m_reply = nullptr;
    QFile* m_outFile = nullptr;
    QString m_tmpPath;
    QString m_destPath;
    qint64 m_minValidBytes = 0;
    qint64 m_expectedSize = 0;
    QByteArray m_expectedSha256;
    QCryptographicHash* m_hash = nullptr;  // streamed while downloading
    bool m_requireGgufMagic = true;
    bool m_busy = false;
};
