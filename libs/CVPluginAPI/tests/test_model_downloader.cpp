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
#include <QFileInfo>
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
    bool ignoreRange = false;
    bool oversizeLength = false;
    qint64 dropAfter = -1;
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
                        // QByteArray::split() is Qt 5.14+; scan with
                        // indexOf to stay on the project's Qt 5.12 floor.
                        int from = 0;
                        while (true) {
                            const int nl = header->indexOf('\n', from);
                            if (nl < 0) break;
                            const QByteArray line =
                                    header->mid(from, nl - from);
                            from = nl + 1;
                            if (line.toLower().startsWith("range: bytes=")) {
                                const QByteArray range = line.mid(13);
                                const int dash = range.indexOf('-');
                                offset = (dash >= 0 ? range.left(dash) : range)
                                                 .toLongLong();
                                ++resumed;
                            }
                        }
                        // Range-blind server mode: a 200-with-full-body
                        // answer must restart the transfer from scratch
                        // instead of appending onto the stale prefix.
                        if (ignoreRange) offset = 0;
                        if (offset >= served.size()) {
                            socket->write(
                                    "HTTP/1.1 416 Range Not Satisfiable\r\n"
                                    "Content-Length: 0\r\n"
                                    "Connection: close\r\n\r\n");
                            socket->disconnectFromHost();
                            return;
                        }
                        const QByteArray body = served.mid(offset);
                        // Weak-link mode: claim the full remaining length
                        // but deliver only dropAfter bytes, then close —
                        // the client sees a truncated transport it must
                        // resume from the bytes already on disk.
                        const QByteArray delivered =
                                (dropAfter >= 0 && body.size() > dropAfter)
                                        ? body.left(dropAfter)
                                        : body;
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
                        // Oversize mode: claim a terabyte so the client's
                        // volume-capacity check must fail the attempt fast
                        // instead of streaming into a full disk.
                        const qint64 declared =
                                oversizeLength
                                        ? 1024LL * 1024 * 1024 * 1024
                                        : static_cast<qint64>(body.size());
                        response += "Content-Length: " +
                                    QByteArray::number(declared) +
                                    "\r\nConnection: close\r\n\r\n";
                        socket->write(response);
                        socket->write(delivered);
                        socket->disconnectFromHost();
                    });
            QObject::connect(socket, &QTcpSocket::disconnected, socket,
                             &QObject::deleteLater);
        }
    });
    ecvModelDownloader downloader;
    auto transfer = [&](const ecvModelDownloader::Request& request,
                        int timeoutMs = 5000) {
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
        timer.start(timeoutMs);
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
    request.url = QString("http://127.0.0.1:%1/model").arg(server.serverPort());
    // [G2] A no-pin .part prefix is never content-proven: the digest is
    // the resume license, so the partial file is discarded and the
    // transfer restarts without a Range request.
    request.destPath = QDir(root).filePath("unpinned.gguf");
    request.contentAnchor = {};
    request.ingestExactSize = model.size();
    {
        QFile part(request.destPath + ".part");
        REQUIRE(part.open(QIODevice::WriteOnly));
        REQUIRE(part.write(model.left(32768)) == 32768);
    }
    const int resumesBeforeUnpinned = resumed;
    REQUIRE(transfer(request));
    REQUIRE(resumed == resumesBeforeUnpinned);
    {
        QFile done(request.destPath);
        REQUIRE(done.open(QIODevice::ReadOnly));
        REQUIRE(done.readAll() == model);
    }
    // [G3] A .part that already holds the complete pinned artifact
    // finalizes locally with zero transport.
    request.destPath = QDir(root).filePath("prefinalized.gguf");
    request.contentAnchor = {
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(model, QCryptographicHash::Sha256)
                    .toHex()};
    request.ingestExactSize = 0;
    {
        QFile part(request.destPath + ".part");
        REQUIRE(part.open(QIODevice::WriteOnly));
        REQUIRE(part.write(model) == model.size());
    }
    const int requestsBeforePrefinalized = requests;
    REQUIRE(transfer(request));
    REQUIRE(requests == requestsBeforePrefinalized);
    REQUIRE(QFileInfo(request.destPath).size() == model.size());
    REQUIRE(!QFile::exists(request.destPath + ".part"));
    // [G3] An oversized .part draws 416; the part is dropped and the next
    // attempt restarts cleanly.
    request.destPath = QDir(root).filePath("oversized.gguf");
    {
        QFile part(request.destPath + ".part");
        REQUIRE(part.open(QIODevice::WriteOnly));
        REQUIRE(part.write(model) == model.size());
        REQUIRE(part.write(QByteArray(4096, 'j')) == 4096);
    }
    REQUIRE(transfer(request));
    REQUIRE(QFileInfo(request.destPath).size() == model.size());
    REQUIRE(!QFile::exists(request.destPath + ".part"));
    // [G3] A rename failure retries the local finalize without destroying
    // the digest-proven part: a blocking directory at the destination
    // fails finalize once, then the retry completes after it is removed.
    request.destPath = QDir(root).filePath("blocked-finalize.gguf");
    REQUIRE(QDir(root).mkdir(QStringLiteral("blocked-finalize.gguf")));
    {
        QFile part(request.destPath + ".part");
        REQUIRE(part.open(QIODevice::WriteOnly));
        REQUIRE(part.write(model) == model.size());
    }
    {
        QEventLoop blockedLoop;
        const auto blockedConnection = QObject::connect(
                &downloader, &ecvModelDownloader::logMessage, &blockedLoop,
                [&](const QString& message) {
                    if (message.contains("Failed to finalize")) {
                        QDir(root).rmdir(
                                QStringLiteral("blocked-finalize.gguf"));
                    }
                });
        REQUIRE(transfer(request, 30000));
        QObject::disconnect(blockedConnection);
    }
    REQUIRE(QFileInfo(request.destPath).isFile());
    REQUIRE(QFileInfo(request.destPath).size() == model.size());
    // [G4] A server that ignores the resume range answers 200 with the
    // full body; the downloader must restart from scratch instead of
    // appending the body onto the stale prefix.
    request.destPath = QDir(root).filePath("range-blind.gguf");
    {
        QFile part(request.destPath + ".part");
        REQUIRE(part.open(QIODevice::WriteOnly));
        REQUIRE(part.write(model.left(32768)) == 32768);
    }
    ignoreRange = true;
    const int resumesBeforeBlind = resumed;
    REQUIRE(transfer(request));
    REQUIRE(resumed == resumesBeforeBlind + 1);
    {
        QFile done(request.destPath);
        REQUIRE(done.open(QIODevice::ReadOnly));
        REQUIRE(done.readAll() == model);
    }
    ignoreRange = false;
    // [G1] Weak-network budget: every interrupted attempt delivers fresh
    // bytes, so the attempt budget must refresh on progress. Without the
    // refresh, 6 attempts x 4 MiB = 24 MiB < 32 MiB would hard-fail.
    served = QByteArray("GGUF") + QByteArray(32 * 1024 * 1024, 'y');
    request.destPath = QDir(root).filePath("weaknet.gguf");
    request.contentAnchor = {
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(served, QCryptographicHash::Sha256)
                    .toHex()};
    dropAfter = 4 * 1024 * 1024;
    {
        bool refreshed = false;
        const auto weakConnection = QObject::connect(
                &downloader, &ecvModelDownloader::logMessage, &app,
                [&](const QString& message) {
                    if (message.contains("attempt budget refreshed"))
                        refreshed = true;
                });
        REQUIRE(transfer(request, 120000));
        QObject::disconnect(weakConnection);
        REQUIRE(refreshed);
    }
    dropAfter = -1;
    REQUIRE(QFileInfo(request.destPath).size() == served.size());
    served = model;
    // [Disk] A declared transfer larger than the volume must fail fast
    // with a clear message instead of streaming into a full disk.
    oversizeLength = true;
    request.destPath = QDir(root).filePath("diskfull.gguf");
    request.contentAnchor = {
            QCryptographicHash::Sha256,
            QCryptographicHash::hash(model, QCryptographicHash::Sha256)
                    .toHex()};
    {
        QEventLoop diskLoop;
        bool diskFail = false;
        const auto diskConnection = QObject::connect(
                &downloader, &ecvModelDownloader::logMessage, &diskLoop,
                [&](const QString& message) {
                    if (message.contains("Insufficient disk space")) {
                        diskFail = true;
                        diskLoop.quit();
                    }
                });
        QTimer::singleShot(30000, &diskLoop, &QEventLoop::quit);
        downloader.download(request);
        diskLoop.exec();
        QObject::disconnect(diskConnection);
        REQUIRE(diskFail);
    }
    downloader.cancel();
    REQUIRE(!downloader.isBusy());
    REQUIRE(!QFile::exists(request.destPath));
    oversizeLength = false;
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
            "retry cancellation, resume, resume license, fast finalize, "
            "weak-network budget, ZIP reuse and UTF-8 paths passed");
    return 0;
}
