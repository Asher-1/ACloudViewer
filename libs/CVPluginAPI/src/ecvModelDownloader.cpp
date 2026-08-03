// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvModelDownloader.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSslConfiguration>
#include <QSslError>
#include <QSslSocket>

ecvModelDownloader::ecvModelDownloader(QObject* parent) : QObject(parent) {
    m_net = new QNetworkAccessManager(this);
}

ecvModelDownloader::~ecvModelDownloader() { cancel(); }

bool ecvModelDownloader::isValidCachedFile(const QString& path,
                                           qint64 minBytes) {
    const QFileInfo fi(path);
    return fi.isFile() && fi.size() >= minBytes;
}

void ecvModelDownloader::removeInvalidCacheFile(const QString& path,
                                                qint64 minBytes) {
    const QFileInfo fi(path);
    if (fi.exists() && fi.size() < minBytes) {
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
    m_minValidBytes =
            request.minValidBytes > 0 ? request.minValidBytes : 1024 * 1024;
    m_tmpPath = m_destPath + QStringLiteral(".part");

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
            m_outFile->write(m_reply->readAll());
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
            } else if (!isValidCachedFile(m_destPath, m_minValidBytes)) {
                emit logMessage(
                        tr("[Download] File too small after download: %1")
                                .arg(m_destPath));
                QFile::remove(m_destPath);
                ok = false;
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
