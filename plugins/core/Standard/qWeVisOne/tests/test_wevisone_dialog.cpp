// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <FileSystem.h>

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkProxy>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QTemporaryDir>
#include <cstdio>

#include "WeVisOneDialog.h"
#include "ecvAssetIntegrity.h"
#include "ecvTestDataRepository.h"

#define REQUIRE(condition)                                               \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTemporaryDir temporary;
    REQUIRE(temporary.isValid());
    REQUIRE(cloudViewer::utility::filesystem::SetEnvVar(
            "CLOUDVIEWER_DATA_ROOT", temporary.path().toUtf8().toStdString()));
    using Repository = ecvTestDataRepository;
    constexpr auto kind = Repository::Dataset::WeVisOne;
    auto& repository = Repository::instance();
    // Synthetic stored ZIP: wevisone_vl_test_data/{livingroom,seabreeze,
    // dogs}.jpg, each a 1x1 PPM payload (the extraction chain is
    // content-agnostic; the pinned-digest cold chain is covered by
    // test_qwevisone_testdata against the real archive).
    const QByteArray bytes = QByteArray::fromHex(
            "504b03041400000000000000215c86f687660e0000000e000000240000007765"
            "7669736f6e655f766c5f746573745f646174612f6c6976696e67726f6f6d2e6a"
            "706750360a3120310a3235350aff0000504b03041400000000000000215c86f6"
            "87660e0000000e0000002300000077657669736f6e655f766c5f746573745f64"
            "6174612f736561627265657a652e6a706750360a3120310a3235350aff000050"
            "4b03041400000000000000215c86f687660e0000000e0000001e000000776576"
            "69736f6e655f766c5f746573745f646174612f646f67732e6a706750360a3120"
            "310a3235350aff0000504b010214031400000000000000215c86f687660e0000"
            "000e00000024000000000000000000000080010000000077657669736f6e655f"
            "766c5f746573745f646174612f6c6976696e67726f6f6d2e6a7067504b010214"
            "031400000000000000215c86f687660e0000000e000000230000000000000000"
            "00000080015000000077657669736f6e655f766c5f746573745f646174612f73"
            "6561627265657a652e6a7067504b010214031400000000000000215c86f68766"
            "0e0000000e0000001e000000000000000000000080019f00000077657669736f"
            "6e655f766c5f746573745f646174612f646f67732e6a7067504b050600000000"
            "03000300ef000000e90000000000");
    const QString archive = Repository::zipPath(kind);
    REQUIRE(QDir().mkpath(QFileInfo(archive).absolutePath()));
    QFile file(archive);
    REQUIRE(file.open(QIODevice::WriteOnly));
    REQUIRE(file.write(bytes) == bytes.size());
    file.close();
    REQUIRE(ecvAssetIntegrity::markVerified(
            archive, Repository::getDatasetInfo(kind).anchor));
    REQUIRE(!QDir(Repository::extractPath(kind)).exists());

    WeVisOneDialog dialog;
    auto* input = dialog.findChild<QLineEdit*>("inputImage");
    auto* sample = dialog.findChild<QPushButton*>("sampleData");
    auto* task = dialog.findChild<QComboBox*>("taskMode");
    auto* query = dialog.findChild<QPlainTextEdit*>("queryText");
    auto* refBoxes = dialog.findChild<QLineEdit*>("refBoxes");
    REQUIRE(input && sample && task && query && refBoxes);
    int completions = 0;
    QObject::connect(&repository, &Repository::datasetReady, &app,
                     [&](bool, Repository::Dataset readyKind) {
                         if (readyKind == kind) ++completions;
                     });
    // Default task (detection): the multi-object living room plus the
    // manifest-verbatim newline-separated category list.
    sample->click();
    const QString demo = Repository::findDatasetFile(kind, "livingroom.jpg");
    REQUIRE(!demo.isEmpty() && input->text() == demo);
    REQUIRE(query->toPlainText() == "sofa\npillow\nlamp");
    REQUIRE(completions == 1 && sample->isEnabled());
    input->clear();
    sample->click();
    REQUIRE(input->text() == demo && completions == 1);

    // Task switch re-fills the task-optimal demo member and its query.
    task->setCurrentIndex(task->findText("referring"));
    sample->click();
    const QString dogs = Repository::findDatasetFile(kind, "dogs.jpg");
    REQUIRE(!dogs.isEmpty() && input->text() == dogs);
    REQUIRE(query->toPlainText() == "the dog closest to the camera");

    // visual_prompt: the seabreeze demo carries the manifest reference box.
    task->setCurrentIndex(task->findText("visual_prompt"));
    sample->click();
    const QString seabreeze =
            Repository::findDatasetFile(kind, "seabreeze.jpg");
    REQUIRE(!seabreeze.isEmpty() && input->text() == seabreeze);
    REQUIRE(refBoxes->text() == "880,120,1500,1070");
    // Back to detection: the previous task's boxes must not leak.
    task->setCurrentIndex(task->findText("detection"));
    sample->click();
    REQUIRE(input->text() == demo && refBoxes->text().isEmpty());

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
            "WeVisOne: cached ZIP fill, task demos, warm reuse and stale "
            "callbacks passed");
    return 0;
}