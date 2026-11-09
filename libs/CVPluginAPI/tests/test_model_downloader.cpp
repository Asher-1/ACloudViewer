// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTimer>
#include <cstdio>
#include <memory>

#include "ecvModelDownloader.h"
#include "ecvTestDataRepository.h"

#define REQUIRE(condition)                                               \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir temporary;
    REQUIRE(temporary.isValid());
    const QString root = temporary.filePath(
            QString::fromUtf8("cache \xe6\xa8\xa1\xe5\x9e\x8b"));
    REQUIRE(QDir().mkpath(root));
    const QByteArray model = QByteArray("GGUF") + QByteArray(96 * 1024, 'x');
    QByteArray served = model;
    int requests = 0;
    int resumed = 0;
    bool serviceUnavailable = false;
    QTcpServer server;
    REQUIRE(server.listen(QHostAddress::LocalHost));
    QObject::connect(&server, &QTcpServer::newConnection, &app, [&]() {
        while (auto* socket = server.nextPendingConnection()) {
            auto header = std::make_shared<QByteArray>();
            QObject::connect(
                    socket, &QTcpSocket::readyRead, socket,
                    [&, socket, header]() {
                        *header += socket->readAll();
                        if (!header->contains("\r\n\r\n")) return;
                        ++requests;
                        if (serviceUnavailable) {
                            socket->write(
                                    "HTTP/1.1 503 Service Unavailable\r\n"
                                    "Content-Length: 0\r\n"
                                    "Connection: close\r\n\r\n");
                            socket->disconnectFromHost();
                            return;
                        }
                        qint64 offset = 0;
                        for (const auto& line : header->split('\n')) {
                            if (line.toLower().startsWith("range: bytes=")) {
                                offset = line.mid(13)
                                                 .split('-')
                                                 .first()
                                                 .toLongLong();
                                ++resumed;
                            }
                        }
                        const QByteArray body = served.mid(offset);
                        QByteArray response =
                                offset > 0 ? "HTTP/1.1 206 Partial Content\r\n"
                                           : "HTTP/1.1 200 OK\r\n";
                        if (offset > 0) {
                            response += "Content-Range: bytes " +
                                        QByteArray::number(offset) + "-" +
                                        QByteArray::number(served.size() - 1) +
                                        "/" +
                                        QByteArray::number(served.size()) +
                                        "\r\n";
                        }
                        response += "Content-Length: " +
                                    QByteArray::number(body.size()) +
                                    "\r\nConnection: close\r\n\r\n" + body;
                        socket->write(response);
                        socket->disconnectFromHost();
                    });
            QObject::connect(socket, &QTcpSocket::disconnected, socket,
                             &QObject::deleteLater);
        }
    });
    ecvModelDownloader downloader;
    auto transfer = [&](const ecvModelDownloader::Request& request) {
        QEventLoop loop;
        QTimer timer;
        timer.setSingleShot(true);
        bool completed = false;
        bool success = false;
        auto connection =
                QObject::connect(&downloader, &ecvModelDownloader::finished,
                                 &loop, [&](bool ok, const QString& path) {
                                     completed = true;
                                     success = ok && path == request.destPath &&
                                               !downloader.isBusy();
                                     loop.quit();
                                 });
        QObject::connect(&timer, &QTimer::timeout, &loop, &QEventLoop::quit);
        timer.start(5000);
        downloader.download(request);
        if (!completed) loop.exec();
        QObject::disconnect(connection);
        if (!completed) downloader.cancel();
        return completed && success;
    };
    ecvModelDownloader::Request request;
    request.url = QString("http://127.0.0.1:%1/model").arg(server.serverPort());
    request.destPath = QDir(root).filePath("selected-q8.gguf");
    request.contentAnchor = {
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(model, QCryptographicHash::Sha256)
                    .toHex()};
    REQUIRE(transfer(request));
    REQUIRE(requests == 1);
    REQUIRE(ecvAssetIntegrity::isModelVerified(request.destPath));
    REQUIRE(transfer(request));
    REQUIRE(transfer(request));
    REQUIRE(requests == 1);
    REQUIRE(!QFile::exists(QDir(root).filePath("selected-f16.gguf")));
    // The shared hit must survive recreating the downloader (new
    // dialog/process).
    downloader.cancel();
    ecvModelDownloader another;
    bool persistentHit = false;
    QObject::connect(&another, &ecvModelDownloader::finished, &app,
                     [&](bool ok, const QString&) { persistentHit = ok; });
    another.download(request);
    REQUIRE(persistentHit && requests == 1);
    // A changed deployed file must be downloaded again, never stat-trusted.
    {
        QFile file(request.destPath);
        REQUIRE(file.open(QIODevice::Append));
        REQUIRE(file.write("changed") == 7);
    }
    REQUIRE(!ecvAssetIntegrity::isModelVerified(request.destPath));
    REQUIRE(transfer(request));
    REQUIRE(requests == 2);
    // Failure must leave the downloader usable for the same selection.
    request.destPath = QDir(root).filePath("second-q8.gguf");
    served = QByteArray("GGUF") + QByteArray(96 * 1024, 'z');
    REQUIRE(!transfer(request));
    REQUIRE(!downloader.isBusy());
    served = model;
    REQUIRE(transfer(request));
    const int afterRetry = requests;
    REQUIRE(transfer(request) && requests == afterRetry);
    // Resume a partial selected model through the real Range transport.
    request.destPath = QDir(root).filePath("resumed-q8.gguf");
    {
        QFile part(request.destPath + ".part");
        REQUIRE(part.open(QIODevice::WriteOnly));
        REQUIRE(part.write(model.left(32768)) == 32768);
    }
    REQUIRE(transfer(request));
    REQUIRE(resumed == 1);
    REQUIRE(transfer(request));
    // An unpinned asset still uses its exact catalog size on first ingestion.
    request.destPath = QDir(root).filePath("sized.gguf");
    request.contentAnchor = {};
    request.ingestExactSize = model.size();
    REQUIRE(transfer(request));
    const int afterSized = requests;
    request.url.clear();  // offline cache hit
    REQUIRE(transfer(request));
    REQUIRE(requests == afterSized);
    // A ZIP uses the same real HTTP transport and digest ledger. Extraction
    // is the repository service's responsibility; no plugin unzip code.
    served = QByteArray::fromHex(
            "504b03041400000000007964695d004b8748140000001400000015000000657861"
            "6d706c"
            "65732f73656c65637465642e74787473656c65637465642073616d706c65206f6e"
            "6c7950"
            "4b010214031400000000007964695d004b87481400000014000000150000000000"
            "000000"
            "0000008001000000006578616d706c65732f73656c65637465642e747874504b05"
            "060000"
            "00000100010043000000470000000000");
    request.url =
            QString("http://127.0.0.1:%1/archive").arg(server.serverPort());
    request.destPath = QDir(root).filePath("samples.zip");
    request.minBytes = 0;
    request.requireGgufMagic = false;
    request.ingestExactSize = 0;
    request.contentAnchor = {
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(served, QCryptographicHash::Sha256)
                    .toHex()};
    REQUIRE(transfer(request));
    const int afterZip = requests;
    REQUIRE(transfer(request));
    REQUIRE(requests == afterZip);
    const QString extracted = QDir(root).filePath("extract");
    REQUIRE(ecvTestDataRepository::zipEntryCount(request.destPath) == 1);
    REQUIRE(ecvTestDataRepository::extractZip(request.destPath, extracted));
    QFile member(QDir(extracted).filePath("examples/selected.txt"));
    REQUIRE(member.open(QIODevice::ReadOnly));
    REQUIRE(member.readAll() == "selected sample only");
    // Backoff belongs to the original request: it is busy, cannot be
    // replaced by another download, and cancellation stops its timer.
    served = model;
    serviceUnavailable = true;
    request.destPath = QDir(root).filePath("cancelled-backoff.gguf");
    request.requireGgufMagic = true;
    request.contentAnchor = {
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(model, QCryptographicHash::Sha256)
                    .toHex()};
    bool backoff = false;
    bool busyAtNotification = false;
    QEventLoop retryLoop;
    const auto retryConnection =
            QObject::connect(&downloader, &ecvModelDownloader::logMessage,
                             &retryLoop, [&](const QString& message) {
                                 if (message.contains("retrying in")) {
                                     backoff = true;
                                     busyAtNotification = downloader.isBusy();
                                     retryLoop.quit();
                                 }
                             });
    QTimer::singleShot(5000, &retryLoop, &QEventLoop::quit);
    downloader.download(request);
    retryLoop.exec();
    QObject::disconnect(retryConnection);
    REQUIRE(backoff && busyAtNotification);
    REQUIRE(downloader.isBusy());
    const int beforeCancel = requests;
    auto replacement = request;
    replacement.destPath = QDir(root).filePath("replacement.gguf");
    downloader.download(replacement);
    int cancelled = 0;
    const auto cancelConnection =
            QObject::connect(&downloader, &ecvModelDownloader::finished, &app,
                             [&](bool ok, const QString& path) {
                                 if (!ok && path == request.destPath)
                                     ++cancelled;
                             });
    downloader.cancel();
    REQUIRE(cancelled == 1 && !downloader.isBusy());
    QEventLoop cancelledLoop;
    QTimer::singleShot(4500, &cancelledLoop, &QEventLoop::quit);
    cancelledLoop.exec();
    REQUIRE(requests == beforeCancel && cancelled == 1);
    REQUIRE(!QFile::exists(replacement.destPath));
    QObject::disconnect(cancelConnection);
    serviceUnavailable = false;
    REQUIRE(transfer(request));
    REQUIRE(transfer(request));
    REQUIRE(requests == beforeCancel + 1);
    std::puts(
            "model downloader: cache hits, selection isolation, corruption, "
            "retry cancellation, resume, ZIP reuse and UTF-8 paths passed");
    return 0;
}
