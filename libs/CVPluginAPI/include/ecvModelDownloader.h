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
#include <QStringList>
#include <QTimer>
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
        // Alternate URLs tried in rotation when the primary endpoint yields
        // no bytes (github release CDN endpoints are unreachable on some
        // networks). Content safety does not depend on the mirror: the
        // pinned digest (or exact size) still gates finalize, and a failed
        // verification deletes the artifact.
        QStringList mirrorUrls;
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
    void startAttempt();
    void scheduleRetry(const QString& reason);
    void finishAttempt(bool ok);

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

    // ── Resume + retry state ────────────────────────────────────────────
    // Large assets over lossy links stall mid-transfer. The .part file is
    // KEPT across failures and each attempt resumes it with an HTTP Range
    // request; the streamed digest covers the full file (the already-
    // downloaded prefix is hashed once when the attempt opens the part in
    // append mode), so resuming can never produce a silently corrupted
    // artifact. Consecutive failed attempts rotate through the URL
    // candidates (primary + mirrors): resume stays valid across mirrors
    // because the candidates serve identical, digest-pinned content.
    ecvModelDownloader::Request m_request;
    QStringList m_candidates;
    int m_candidateIndex = 0;
    qint64 m_resumeOffset = 0; /**< bytes already on disk for attempt */
    int m_attempt = 0;         /**< 0-based attempt counter */
    qint64 m_lastProgressBytes = -1;
    qint64 m_lastWatchdogBytes = -1;
    qint64 m_fullTotal = 0; /**< resume prefix + reply total */
    int m_stallTicks = 0;
    QTimer* m_stallTimer = nullptr; /**< no-progress watchdog */
    QTimer* m_retryTimer = nullptr; /**< backoff between attempts */
    bool m_rangeNegotiated = false; /**< 206 verified for this attempt */

    double progressPercent() const;

    static constexpr int kMaxAttempts = 6;
    static constexpr int kStallCheckMs = 5000;
    static constexpr int kStallTimeoutMs = 30000;
    static constexpr int kRetryDelayBaseMs = 2000;
};
