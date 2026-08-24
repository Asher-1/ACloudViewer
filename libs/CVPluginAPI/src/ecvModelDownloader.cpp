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
#include <cstring>

ecvModelDownloader::ecvModelDownloader(QObject* parent) : QObject(parent) {
    m_net = new QNetworkAccessManager(this);
}

ecvModelDownloader::~ecvModelDownloader() { cancel(); }

// GGUF file format magic. The first 4 bytes of every valid GGUF file are
// the ASCII characters "GGUF" (0x46475547 in little-endian). Validating
// against this magic — instead of guessing a per-model size floor — lets
// us accept the smallest quantized ALIKED extractor (~714 KiB) while
// still rejecting truncated/empty/HTML responses.
static constexpr const char kGgufMagic[4] = {'G', 'G', 'U', 'F'};

static bool hasGgufMagic(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return false;
    char head[4] = {0, 0, 0, 0};
    const qint64 read = f.read(head, sizeof(head));
    f.close();
    if (read != sizeof(head)) return false;
    return std::memcmp(head, kGgufMagic, sizeof(kGgufMagic)) == 0;
}

// Full-file SHA-256 (hex). Reads the whole file — callers must only use this
// for one-shot verification, never for per-dialog checks on multi-GB models.
static QByteArray sha256OfFile(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&f)) return {};
    f.close();
    return hash.result().toHex();
}

bool ecvModelDownloader::isValidCachedFile(const QString& path,
                                           qint64 minBytes,
                                           bool requireGgufMagic,
                                           qint64 expectedSize,
                                           const QByteArray& expectedSha256) {
    const QFileInfo fi(path);
    if (!fi.isFile() || fi.size() < minBytes) return false;
    if (expectedSize > 0 && fi.size() != expectedSize) return false;
    if (requireGgufMagic && !hasGgufMagic(path)) return false;
    if (!expectedSha256.isEmpty() && sha256OfFile(path) != expectedSha256) {
        return false;
    }
    return true;
}

void ecvModelDownloader::removeInvalidCacheFile(
        const QString& path,
        qint64 minBytes,
        bool requireGgufMagic,
        qint64 expectedSize,
        const QByteArray& expectedSha256) {
    if (!isValidCachedFile(path, minBytes, requireGgufMagic, expectedSize,
                           expectedSha256)) {
        QFile::remove(path);
    }
}

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
    if (!m_tmpPath.isEmpty()) {
        QFile::remove(m_tmpPath);
        m_tmpPath.clear();
    }
    m_busy = false;
}

void ecvModelDownloader::cancel() {
    if (!m_busy) return;
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

    m_destPath = request.destPath;
    m_minValidBytes = request.minBytes > 0 ? request.minBytes : 64 * 1024;
    m_expectedSize = request.expectedSize;
    m_expectedSha256 = request.expectedSha256;
    m_requireGgufMagic = request.requireGgufMagic;
    m_tmpPath = m_destPath + QStringLiteral(".part");
    // Stream the SHA-256 while writing: for multi-GB models this avoids a
    // second full read pass that a post-download hash check would need.
    m_hash = new QCryptographicHash(QCryptographicHash::Sha256);

    QDir().mkpath(QFileInfo(m_destPath).absolutePath());
    QFile::remove(m_tmpPath);

    QNetworkRequest req{QUrl(request.url)};
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                     QNetworkRequest::NoLessSafeRedirectPolicy);
    auto sslConfig = QSslConfiguration::defaultConfiguration();
    sslConfig.setPeerVerifyMode(QSslSocket::VerifyNone);
    req.setSslConfiguration(sslConfig);

    m_reply = m_net->get(req);
    m_busy = true;

    connect(m_reply, &QNetworkReply::sslErrors, this,
            [this](const QList<QSslError>& errors) {
                for (const auto& e : errors) {
                    emit logMessage(tr("[Download] SSL warning (ignored): %1")
                                            .arg(e.errorString()));
                }
                if (m_reply) m_reply->ignoreSslErrors();
            });

    m_outFile = new QFile(m_tmpPath, this);
    if (!m_outFile->open(QIODevice::WriteOnly)) {
        emit logMessage(tr("[Download] Cannot write to %1").arg(m_tmpPath));
        cleanupActiveReply();
        emit finished(false, m_destPath);
        return;
    }

    connect(m_reply, &QNetworkReply::readyRead, this, [this]() {
        if (m_outFile && m_reply) {
            const QByteArray chunk = m_reply->readAll();
            m_outFile->write(chunk);
            if (m_hash) {
                m_hash->addData(chunk);
            }
        }
    });
    connect(m_reply, &QNetworkReply::downloadProgress, this,
            &ecvModelDownloader::progress);
    connect(m_reply, &QNetworkReply::finished, this, [this]() {
        if (m_outFile) {
            m_outFile->close();
            m_outFile->deleteLater();
            m_outFile = nullptr;
        }

        bool ok = m_reply && m_reply->error() == QNetworkReply::NoError;
        if (ok) {
            QFile::remove(m_destPath);
            ok = QFile::rename(m_tmpPath, m_destPath);
            if (!ok) {
                emit logMessage(
                        tr("[Download] Failed to finalize %1").arg(m_destPath));
            } else if (!isValidCachedFile(m_destPath, m_minValidBytes,
                                          m_requireGgufMagic, m_expectedSize)) {
                // Surface the actual reason (too small vs wrong magic vs size
                // mismatch) so operators can distinguish a truncated
                // connection from a genuine 200-with-wrong-content response
                // (e.g. a captive portal HTML page masquerading as the model,
                // or a CDN error body that still starts with the GGUF magic).
                const QFileInfo fi(m_destPath);
                if (!fi.exists()) {
                    emit logMessage(tr("[Download] Output file missing after "
                                       "rename: %1")
                                            .arg(m_destPath));
                } else if (fi.size() < m_minValidBytes) {
                    emit logMessage(
                            tr("[Download] File too small after download: "
                               "%1 (%2 bytes, need >= %3)")
                                    .arg(m_destPath)
                                    .arg(fi.size())
                                    .arg(m_minValidBytes));
                } else if (m_expectedSize > 0 && fi.size() != m_expectedSize) {
                    emit logMessage(tr("[Download] File size mismatch after "
                                       "download: %1 (%2 bytes, expected "
                                       "%3)")
                                            .arg(m_destPath)
                                            .arg(fi.size())
                                            .arg(m_expectedSize));
                } else {
                    emit logMessage(tr("[Download] File lacks GGUF magic: %1")
                                            .arg(m_destPath));
                }
                QFile::remove(m_destPath);
                ok = false;
            } else if (!m_expectedSha256.isEmpty()) {
                // Content-level check: the hash was streamed while writing,
                // so no second read pass is needed.
                const QByteArray actual =
                        m_hash ? m_hash->result().toHex() : QByteArray();
                if (actual.compare(m_expectedSha256, Qt::CaseInsensitive) !=
                    0) {
                    emit logMessage(
                            tr("[Download] SHA-256 mismatch after download: "
                               "%1 (got %2, expected %3)")
                                    .arg(m_destPath)
                                    .arg(QString::fromLatin1(actual))
                                    .arg(QString::fromLatin1(
                                            m_expectedSha256)));
                    QFile::remove(m_destPath);
                    ok = false;
                }
            }
        } else if (m_reply) {
            emit logMessage(
                    tr("[Download] Failed: %1").arg(m_reply->errorString()));
        }

        if (!ok && !m_tmpPath.isEmpty()) {
            QFile::remove(m_tmpPath);
        }

        const QString dest = m_destPath;
        cleanupActiveReply();
        emit finished(ok, dest);
    });
}
