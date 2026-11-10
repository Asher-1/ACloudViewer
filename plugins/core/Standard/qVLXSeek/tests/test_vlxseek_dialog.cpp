// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <FileSystem.h>

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QPushButton>
#include <QTemporaryDir>
#include <cstdio>

#include "VLXSeekDialog.h"
#include "ecvAssetIntegrity.h"
#include "ecvTestDataRepository.h"

#define REQUIRE(condition)                                               \
    do {                                                                 \
        std::fprintf(stderr, "checkpoint line %d\n", __LINE__); \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    QApplication app(argc, argv);
    QTemporaryDir temporary;
    REQUIRE(temporary.isValid());
    REQUIRE(cloudViewer::utility::filesystem::SetEnvVar(
            "CLOUDVIEWER_DATA_ROOT", temporary.path().toUtf8().toStdString()));
    using Repository = ecvTestDataRepository;
    constexpr auto kind = Repository::Dataset::VlxSeek;
    auto& repository = Repository::instance();
    const QByteArray bytes = QByteArray::fromHex(
            "504b03041400000000000000215c86f687660e0000000e00000023000000766c78"
            "5f7365656b"
            "5f746573745f696d616765732f64656d6f5f696d6167652e6a706750360a312031"
            "0a3235350a"
            "ff0000504b03041400000000000000215c86f687660e0000000e00000024000000"
            "766c785f73"
            "65656b5f746573745f696d616765732f64656d6f5f696d616765322e6a70675036"
            "0a3120310a"
            "3235350aff0000504b03041400000000000000215c86f687660e0000000e000000"
            "2500000076"
            "6c785f7365656b5f746573745f696d616765732f6562696b655f7269646572732e"
            "6a70675036"
            "0a3120310a3235350aff0000504b010214031400000000000000215c86f687660e"
            "0000000e00"
            "0000230000000000000000000000800100000000766c785f7365656b5f74657374"
            "5f696d6167"
            "65732f64656d6f5f696d6167652e6a7067504b010214031400000000000000215c"
            "86f687660e"
            "0000000e00000024000000000000000000000080014f000000766c785f7365656b"
            "5f74657374"
            "5f696d616765732f64656d6f5f696d616765322e6a7067504b0102140314000000"
            "0000000021"
            "5c86f687660e0000000e00000025000000000000000000000080019f000000766c"
            "785f736565"
            "6b5f746573745f696d616765732f6562696b655f7269646572732e6a7067504b05"
            "0600000000"
            "03000300f6000000f00000000000");
    const QString archive = Repository::zipPath(kind);
    REQUIRE(QDir().mkpath(QFileInfo(archive).absolutePath()));
    QFile file(archive);
    REQUIRE(file.open(QIODevice::WriteOnly));
    REQUIRE(file.write(bytes) == bytes.size());
    file.close();
    REQUIRE(ecvAssetIntegrity::markVerified(
            archive, Repository::getDatasetInfo(kind).anchor));
    REQUIRE(!QDir(Repository::extractPath(kind)).exists());

    VLXSeekDialog dialog;
    auto* input = dialog.findChild<QLineEdit*>("inputImage");
    auto* sample = dialog.findChild<QPushButton*>("sampleData");
    REQUIRE(input && sample);
    int completions = 0;
    QObject::connect(&repository, &Repository::datasetReady, &app,
                     [&](bool, Repository::Dataset readyKind) {
                         if (readyKind == kind) ++completions;
                     });
    sample->click();
    const QString demo = Repository::findDatasetFile(kind, "ebike_riders.jpg");
    REQUIRE(!demo.isEmpty() && input->text() == demo);
    REQUIRE(completions == 1 && sample->isEnabled());
    input->clear();
    sample->click();
    REQUIRE(input->text() == demo && completions == 1);

    // Idle broadcasts cannot overwrite an explicit user input.
    input->setText("user-selected-image.png");
    emit repository.datasetReady(true, kind);
    REQUIRE(input->text() == "user-selected-image.png");

    // Close a pending cold request, then deliver a late completion.
    REQUIRE(QFile::remove(archive));
    REQUIRE(QDir(Repository::extractPath(kind)).removeRecursively());
    auto* manager = repository.findChild<QNetworkAccessManager*>();
    REQUIRE(manager);
    manager->setProxy(QNetworkProxy(QNetworkProxy::HttpProxy, "127.0.0.1", 9));
    sample->click();
    REQUIRE(!sample->isEnabled());
    dialog.reject();
    REQUIRE(sample->isEnabled());
    REQUIRE(QDir().mkpath(QFileInfo(demo).absolutePath()));
    QFile lateDemo(demo);
    REQUIRE(lateDemo.open(QIODevice::WriteOnly));
    REQUIRE(lateDemo.write("P6\n1 1\n255\n\xff\0\0", 14) == 14);
    lateDemo.close();
    emit repository.datasetReady(true, kind);
    REQUIRE(input->text() == "user-selected-image.png");
    repository.cancelDownload();
    std::puts(
            "VLXSeek: cached ZIP fill, warm reuse and stale callbacks passed");
    std::fprintf(stderr, "body done, returning from main\n");
    return 0;
}
