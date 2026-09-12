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
#include "ecvAssetIntegrity.h"

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
        // Validation policy for the downloaded file — INGESTION TIME ONLY.
        // All checks run once at finalize, never on subsequent accesses
        // (verified files are stat-trusted via ecvAssetIntegrity):
        //   1. minBytes floor rejects empty/HTML junk pages;
        //   2. requireGgufMagic accepts only files starting with "GGUF";
        //   3. when contentAnchor carries a digest, the file is stream-hashed
        //      WHILE downloading and compared once at finalize — truncation
        //      and corruption are both caught with no extra read pass;
        //   4. otherwise, ingestExactSize (when > 0) is compared once at
        //      finalize — the single truncation guard for assets without a
        //      pinned digest (e.g. GitHub-release GGUFs). NOT a per-access
        //      policy.
        // minBytes default is 64 KiB: large enough to reject empty/HTML
        // pages, small enough to admit the smallest ALIKED
        // (aliked-n16rot-q8_0.gguf is ~714 KiB on disk).
        qint64 minBytes = 64 * 1024;
        bool requireGgufMagic = true;
        // Content identity pinned in source (algo + hex digest, e.g. the
        // HF LFS oid). Streamed while downloading; a mismatch deletes the
        // file and fails the download.
        ecvAssetIntegrity::Anchor contentAnchor;
        // Exact expected byte count, used ONLY when contentAnchor has no
        // digest (see policy item 4 above).
        qint64 ingestExactSize = 0;
    };

    explicit ecvModelDownloader(QObject* parent = nullptr);
    ~ecvModelDownloader() override;

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
    ecvAssetIntegrity::Anchor m_contentAnchor;
    qint64 m_ingestExactSize = 0;
    QCryptographicHash* m_hash = nullptr;  // streamed while downloading
    bool m_requireGgufMagic = true;
    bool m_busy = false;
};
