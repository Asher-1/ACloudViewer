// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvModelDownloader.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIODevice>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSslConfiguration>
#include <QSslError>
#include <QSslSocket>
#include <QVariant>

ecvModelDownloader::ecvModelDownloader(QObject* parent) : QObject(parent) {
    m_net = new QNetworkAccessManager(this);
    // No-progress watchdog: aborts a stalled transfer so the retry loop can
    // resume it from the bytes already on disk.
    m_stallTimer = new QTimer(this);
    m_stallTimer->setInterval(kStallCheckMs);
    connect(m_stallTimer, &QTimer::timeout, this, [this]() {
        if (!m_busy || !m_reply) return;
        if (m_lastProgressBytes == m_lastWatchdogBytes) {
            ++m_stallTicks;
        } else {
            m_stallTicks = 0;
            m_lastWatchdogBytes = m_lastProgressBytes;
        }
        if (m_stallTicks * kStallCheckMs >= kStallTimeoutMs) {
            m_stallTicks = 0;
            emit logMessage(
                    tr("[Download] No progress for %1 s — aborting "
                       "to resume from %2%")
                            .arg(kStallTimeoutMs / 1000)
                            .arg(QString::number(progressPercent(), 'f', 0)));
            m_reply->abort();  // triggers the finished handler → retry
        }
    });
    m_retryTimer = new QTimer(this);
    m_retryTimer->setSingleShot(true);
    connect(m_retryTimer, &QTimer::timeout, this, [this]() { startAttempt(); });
}

ecvModelDownloader::~ecvModelDownloader() { cancel(); }

QString ecvModelDownloader::formatFileSize(qint64 bytes) {
    if (bytes < 0) {
        return QStringLiteral("?");
    }
    if (bytes < 1024) {
        return QStringLiteral("%1 B").arg(bytes);
    }
    if (bytes < 1024LL * 1024) {
        return QStringLiteral("%1 KB").arg(bytes / 1024.0, 0, 'f', 1);
    }
    if (bytes < 1024LL * 1024 * 1024) {
        return QStringLiteral("%1 MB").arg(bytes / (1024.0 * 1024.0), 0, 'f',
                                           1);
    }
    return QStringLiteral("%1 GB").arg(bytes / (1024.0 * 1024.0 * 1024.0), 0,
                                       'f', 2);
}

QString ecvModelDownloader::formatDownloadProgress(qint64 received,
                                                   qint64 total) {
    if (total > 0) {
        const int percent =
                static_cast<int>(received * 100 / qMax<qint64>(total, 1));
        return QObject::tr("%1 / %2 (%3%)")
                .arg(formatFileSize(received))
                .arg(formatFileSize(total))
                .arg(percent);
    }
    return QObject::tr("%1 downloaded").arg(formatFileSize(received));
}

double ecvModelDownloader::progressPercent() const {
    const qint64 total = m_fullTotal > 0 ? m_fullTotal : 0;
    if (total <= 0) return 0.0;
    return (m_resumeOffset + m_lastProgressBytes) * 100.0 / total;
}

void ecvModelDownloader::cleanupActiveReply() {
    if (m_outFile) {
        m_outFile->close();
        m_outFile->deleteLater();
        m_outFile = nullptr;
    }
    if (m_reply) {
        m_reply->abort();
        m_reply->deleteLater();
        m_reply = nullptr;
    }
    delete m_hash;
    m_hash = nullptr;
    // The .part file is intentionally KEPT: the next attempt (automatic
    // retry or a manual re-click) resumes it with an HTTP Range request.
    m_busy = false;
}

void ecvModelDownloader::cancel() {
    if (!m_busy) return;
    if (m_stallTimer) m_stallTimer->stop();
    if (m_retryTimer) m_retryTimer->stop();
    cleanupActiveReply();
    emit logMessage(tr("[Download] Cancelled."));
}

QUrl ecvModelDownloader::hfDownloadUrl(const QString& repoId,
                                       const QString& filename) {
    return QUrl(QStringLiteral("https://huggingface.co/%1/resolve/main/%2")
                        .arg(repoId, filename));
}

void ecvModelDownloader::download(const Request& request) {
    if (m_busy) {
        emit logMessage(tr("[Download] Already in progress."));
        return;
    }
    if (request.url.isEmpty() || request.destPath.isEmpty()) {
        emit finished(false, request.destPath);
        return;
    }

    m_request = request;
    // URL candidate list: the primary endpoint first, then mirrors. For
    // github.com release assets, append well-known prefix-style proxies
    // (the release CDN is unreachable on some networks while github.com
    // itself is reachable). Integrity does not depend on the mirror — the
    // finalize digest check rejects anything it serves that differs from
    // the pinned artifact.
    m_candidates.clear();
    m_candidates.append(m_request.url);
    if (QUrl(m_request.url).host() == QStringLiteral("github.com")) {
        for (const QString& proxy : {QStringLiteral("https://ghfast.top/"),
                                     QStringLiteral("https://gh-proxy.com/")}) {
            m_candidates.append(proxy + m_request.url);
        }
    }
    for (const QString& mirror : m_request.mirrorUrls) {
        if (!mirror.isEmpty()) m_candidates.append(mirror);
    }
    m_attempt = 0;
    m_candidateIndex = 0;
    startAttempt();
}

void ecvModelDownloader::startAttempt() {
    Q_ASSERT(!m_busy);
    if (m_request.url.isEmpty() || m_request.destPath.isEmpty()) {
        emit finished(false, m_request.destPath);
        return;
    }

    m_destPath = m_request.destPath;
    m_minValidBytes = m_request.minBytes > 0 ? m_request.minBytes : 64 * 1024;
    m_contentAnchor = m_request.contentAnchor;
    m_ingestExactSize = m_request.ingestExactSize;
    m_requireGgufMagic = m_request.requireGgufMagic;
    m_tmpPath = m_destPath + QStringLiteral(".part");
    const QString attemptUrl =
            m_candidates.isEmpty()
                    ? m_request.url
                    : m_candidates.at(m_candidateIndex % m_candidates.size());

    // Resume an interrupted transfer: the .part file keeps the bytes of the
    // previous attempts. The digest covers the FULL file, so the prefix on
    // disk is hashed once here and the incoming remainder is streamed into
    // the same hash — a resumed artifact can never pass the finalize check
    // with a corrupted prefix.
    m_resumeOffset = 0;
    m_rangeNegotiated = false;
    const qint64 partBytes =
            QFileInfo(m_tmpPath).exists() ? QFileInfo(m_tmpPath).size() : 0;
    m_hash = m_contentAnchor.digestHex.isEmpty()
                     ? nullptr
                     : new QCryptographicHash(m_contentAnchor.algo);
    if (partBytes > 0) {
        if (m_hash) {
            QFile part(m_tmpPath);
            if (part.open(QIODevice::ReadOnly)) {
                constexpr qint64 kHashChunk = 1 << 20;
                while (!part.atEnd()) {
                    m_hash->addData(part.read(kHashChunk));
                }
            }
        }
        m_resumeOffset = partBytes;
    }

    QDir().mkpath(QFileInfo(m_destPath).absolutePath());

    QNetworkRequest req{QUrl(attemptUrl)};
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                     QNetworkRequest::NoLessSafeRedirectPolicy);
    auto sslConfig = QSslConfiguration::defaultConfiguration();
    sslConfig.setPeerVerifyMode(QSslSocket::VerifyNone);
    req.setSslConfiguration(sslConfig);
    if (m_resumeOffset > 0) {
        req.setRawHeader(QByteArray("Range"),
                         QByteArray("bytes=") +
                                 QByteArray::number(m_resumeOffset) + '-');
        emit logMessage(tr("[Download] Resuming %1 from %2 (attempt %3/%4)")
                                .arg(QFileInfo(m_destPath).fileName())
                                .arg(formatFileSize(m_resumeOffset))
                                .arg(m_attempt + 1)
                                .arg(kMaxAttempts));
    } else if (m_candidates.size() > 1) {
        emit logMessage(tr("[Download] Trying %1 (attempt %2/%4)")
                                .arg(attemptUrl)
                                .arg(m_attempt + 1)
                                .arg(kMaxAttempts));
    }

    m_reply = m_net->get(req);
    m_busy = true;
    m_lastProgressBytes = 0;
    m_lastWatchdogBytes = -1;
    m_stallTicks = 0;
    m_fullTotal = 0;
    m_stallTimer->start();

    connect(m_reply, &QNetworkReply::sslErrors, this,
            [this](const QList<QSslError>& errors) {
                for (const auto& e : errors) {
                    emit logMessage(tr("[Download] SSL warning (ignored): %1")
                                            .arg(e.errorString()));
                }
                if (m_reply) m_reply->ignoreSslErrors();
            });

    m_outFile = new QFile(m_tmpPath, this);
    if (!m_outFile->open(m_resumeOffset > 0 ? QIODevice::Append
                                            : QIODevice::WriteOnly)) {
        emit logMessage(tr("[Download] Cannot write to %1").arg(m_tmpPath));
        m_stallTimer->stop();
        cleanupActiveReply();
        emit finished(false, m_destPath);
        return;
    }

    // A server that ignores the Range header answers 200 with the FULL
    // body; appending that would corrupt the artifact. Detect the status
    // once from the reply metadata and restart from scratch when needed.
    if (m_resumeOffset > 0) {
        connect(m_reply, &QNetworkReply::metaDataChanged, this, [this]() {
            if (m_rangeNegotiated || !m_reply || !m_outFile) return;
            const QVariant code = m_reply->attribute(
                    QNetworkRequest::HttpStatusCodeAttribute);
            if (!code.isValid()) return;  // not finalized yet
            m_rangeNegotiated = true;
            if (code.toInt() == 200) {
                emit logMessage(
                        tr("[Download] Server ignored the resume "
                           "range — restarting from scratch"));
                m_outFile->close();
                m_outFile->deleteLater();
                m_outFile = nullptr;
                QFile::remove(m_tmpPath);
                delete m_hash;
                m_hash = m_contentAnchor.digestHex.isEmpty()
                                 ? nullptr
                                 : new QCryptographicHash(m_contentAnchor.algo);
                m_resumeOffset = 0;
                m_outFile = new QFile(m_tmpPath, this);
                if (!m_outFile->open(QIODevice::WriteOnly)) {
                    emit logMessage(
                            tr("[Download] Cannot write to %1").arg(m_tmpPath));
                    m_stallTimer->stop();
                    cleanupActiveReply();
                    emit finished(false, m_destPath);
                }
            }
        });
    }

    connect(m_reply, &QNetworkReply::readyRead, this, [this]() {
        if (m_outFile && m_reply) {
            const QByteArray chunk = m_reply->readAll();
            if (chunk.isEmpty()) return;
            m_outFile->write(chunk);
            if (m_hash) {
                m_hash->addData(chunk);
            }
            m_lastProgressBytes += chunk.size();
        }
    });
    connect(m_reply, &QNetworkReply::downloadProgress, this,
            [this](qint64 received, qint64 total) {
                // 206 replies report only the remaining range as total;
                // fold the resumed prefix back in for the UI.
                m_fullTotal = m_resumeOffset + total;
                emit progress(m_resumeOffset + received, m_fullTotal);
            });
    connect(m_reply, &QNetworkReply::finished, this, [this]() {
        finishAttempt(m_reply && m_reply->error() == QNetworkReply::NoError);
    });
}

void ecvModelDownloader::scheduleRetry(const QString& reason) {
    ++m_attempt;
    // Rotate to the next URL candidate each attempt (direct → mirror1 →
    // mirror2 → …) so an unreachable endpoint cannot burn all attempts.
    if (!m_candidates.isEmpty()) {
        const int next = (m_candidateIndex + 1) % m_candidates.size();
        if (next != m_candidateIndex) {
            m_candidateIndex = next;
            emit logMessage(tr("[Download] Switching endpoint: %1")
                                    .arg(m_candidates.at(m_candidateIndex)));
        }
    }
    const int delay = qMin(kRetryDelayBaseMs << qMin(m_attempt, 4), 15000);
    emit logMessage(tr("[Download] %1 — retrying in %2 s (attempt %3/%4)")
                            .arg(reason)
                            .arg(delay / 1000)
                            .arg(m_attempt + 1)
                            .arg(kMaxAttempts));
    m_retryTimer->start(delay);
}

void ecvModelDownloader::finishAttempt(bool transportOk) {
    m_stallTimer->stop();
    if (m_outFile) {
        m_outFile->close();
        m_outFile->deleteLater();
        m_outFile = nullptr;
    }

    bool ok = transportOk;
    QString failureReason;
    bool permanent = false;
    if (!transportOk) {
        failureReason = m_reply ? m_reply->errorString()
                                : tr("unknown transport error");
    }

    if (ok) {
        QFile::remove(m_destPath);
        ok = QFile::rename(m_tmpPath, m_destPath);
        if (!ok) {
            failureReason = tr("failed to finalize %1").arg(m_destPath);
            ecvAssetIntegrity::invalidate(m_destPath);
        } else if (!ecvAssetIntegrity::passesCheapChecks(
                           m_destPath, m_minValidBytes, m_requireGgufMagic)) {
            // Surface the actual reason (too small vs wrong magic) so
            // operators can distinguish a truncated connection from a
            // genuine 200-with-wrong-content response (e.g. a captive
            // portal HTML page masquerading as the model).
            const QFileInfo fi(m_destPath);
            if (!fi.exists()) {
                failureReason = tr("output file missing after rename: %1")
                                        .arg(m_destPath);
            } else if (fi.size() < m_minValidBytes) {
                failureReason = tr("file too small after download: %1 "
                                   "(%2 bytes, need >= %3)")
                                        .arg(m_destPath)
                                        .arg(fi.size())
                                        .arg(m_minValidBytes);
            } else {
                failureReason = tr("file lacks GGUF magic: %1").arg(m_destPath);
            }
            QFile::remove(m_destPath);
            ecvAssetIntegrity::invalidate(m_destPath);
            ok = false;
            permanent = true;
        } else if (!m_contentAnchor.digestHex.isEmpty()) {
            // Content-level check: the digest was streamed while
            // writing (resumed prefix included), so no second read
            // pass is needed.
            const QByteArray actual =
                    m_hash ? m_hash->result().toHex() : QByteArray();
            if (actual.compare(m_contentAnchor.digestHex,
                               Qt::CaseInsensitive) != 0) {
                failureReason = tr("content digest mismatch after "
                                   "download: %1 (got %2, expected %3)")
                                        .arg(m_destPath)
                                        .arg(QString::fromLatin1(actual))
                                        .arg(QString::fromLatin1(
                                                m_contentAnchor.digestHex));
                QFile::remove(m_destPath);
                ecvAssetIntegrity::invalidate(m_destPath);
                ok = false;
                permanent = true;
            }
        } else if (m_ingestExactSize > 0 &&
                   QFileInfo(m_destPath).size() != m_ingestExactSize) {
            failureReason = tr("file size mismatch after download: %1 "
                               "(%2 bytes, expected %3)")
                                    .arg(m_destPath)
                                    .arg(QFileInfo(m_destPath).size())
                                    .arg(m_ingestExactSize);
            QFile::remove(m_destPath);
            ecvAssetIntegrity::invalidate(m_destPath);
            ok = false;
            permanent = true;
        }
        if (ok) {
            // Record the verified state so every later presence check
            // is a stat-only lookup (no re-hashing of multi-GB files).
            ecvAssetIntegrity::markVerified(m_destPath, m_contentAnchor);
        }
    }

    if (!ok && !permanent) {
        // Keep the .part file: the next attempt resumes it.
    } else if (!ok) {
        QFile::remove(m_tmpPath);
    }

    const QString dest = m_destPath;
    cleanupActiveReply();

    if (!ok && !permanent && m_attempt + 1 < kMaxAttempts) {
        emit logMessage(tr("[Download] Failed: %1").arg(failureReason));
        scheduleRetry(failureReason);
        return;
    }
    if (!ok) {
        emit logMessage(tr("[Download] Failed: %1").arg(failureReason));
    }
    emit finished(ok, dest);
}
