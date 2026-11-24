// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <FileSystem.h>
#include <gtest/gtest.h>

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

class VlxSeekDialogTest : public ::testing::Test {
protected:
    using Repository = ecvTestDataRepository;
    static constexpr auto kind = Repository::Dataset::VlxSeek;
    QTemporaryDir temporary;
    Repository& repository = Repository::instance();
    QString archive;
    void SetUp() override {
        ASSERT_TRUE(temporary.isValid());
        ASSERT_TRUE(cloudViewer::utility::filesystem::SetEnvVar(
                "CLOUDVIEWER_DATA_ROOT",
                temporary.path().toUtf8().toStdString()));
        const QByteArray bytes = QByteArray::fromHex(
                "504b03041400000000000000215c86f687660e0000000e0000002300000076"
                "6c78"
                "5f7365656b"
                "5f746573745f696d616765732f64656d6f5f696d6167652e6a706750360a31"
                "2031"
                "0a3235350a"
                "ff0000504b03041400000000000000215c86f687660e0000000e0000002400"
                "0000"
                "766c785f73"
                "65656b5f746573745f696d616765732f64656d6f5f696d616765322e6a7067"
                "5036"
                "0a3120310a"
                "3235350aff0000504b03041400000000000000215c86f687660e0000000e00"
                "0000"
                "2500000076"
                "6c785f7365656b5f746573745f696d616765732f6562696b655f7269646572"
                "732e"
                "6a70675036"
                "0a3120310a3235350aff0000504b010214031400000000000000215c86f687"
                "660e"
                "0000000e00"
                "0000230000000000000000000000800100000000766c785f7365656b5f7465"
                "7374"
                "5f696d6167"
                "65732f64656d6f5f696d6167652e6a7067504b010214031400000000000000"
                "215c"
                "86f687660e"
                "0000000e00000024000000000000000000000080014f000000766c785f7365"
                "656b"
                "5f74657374"
                "5f696d616765732f64656d6f5f696d616765322e6a7067504b010214031400"
                "0000"
                "0000000021"
                "5c86f687660e0000000e00000025000000000000000000000080019f000000"
                "766c"
                "785f736565"
                "6b5f746573745f696d616765732f6562696b655f7269646572732e6a706750"
                "4b05"
                "0600000000"
                "03000300f6000000f00000000000");
        archive = Repository::zipPath(kind);
        ASSERT_TRUE(QDir().mkpath(QFileInfo(archive).absolutePath()));
        QFile file(archive);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        ASSERT_TRUE(file.write(bytes) == bytes.size());
        file.close();
        ASSERT_TRUE(ecvAssetIntegrity::markVerified(
                archive, Repository::getDatasetInfo(kind).anchor));
        ASSERT_TRUE(!QDir(Repository::extractPath(kind)).exists());
    }
    void TearDown() override {
        repository.cancelDownload();
        if (auto* manager = repository.findChild<QNetworkAccessManager*>())
            manager->setProxy(QNetworkProxy::DefaultProxy);
    }
};

TEST_F(VlxSeekDialogTest, CachedZipAutofillsWarmReuseAndIgnoresIdleBroadcasts) {
    VLXSeekDialog dialog;
    auto* input = dialog.findChild<QLineEdit*>("inputImage");
    auto* sample = dialog.findChild<QPushButton*>("sampleData");
    ASSERT_TRUE(input && sample);
    int completions = 0;
    QObject::connect(&repository, &Repository::datasetReady, &dialog,
                     [&](bool, Repository::Dataset readyKind) {
                         if (readyKind == kind) ++completions;
                     });
    sample->click();
    const QString demo = Repository::findDatasetFile(kind, "ebike_riders.jpg");
    ASSERT_TRUE(!demo.isEmpty() && input->text() == demo);
    ASSERT_TRUE(completions == 1 && sample->isEnabled());
    input->clear();
    sample->click();
    ASSERT_TRUE(input->text() == demo && completions == 1);

    // Idle broadcasts cannot overwrite an explicit user input.
    input->setText("user-selected-image.png");
    emit repository.datasetReady(true, kind);
    ASSERT_TRUE(input->text() == "user-selected-image.png");
}

TEST_F(VlxSeekDialogTest, RejectDetachesPendingConsumerFromLateCompletion) {
    VLXSeekDialog dialog;
    auto* input = dialog.findChild<QLineEdit*>("inputImage");
    auto* sample = dialog.findChild<QPushButton*>("sampleData");
    ASSERT_TRUE(input && sample);
    sample->click();
    const QString demo = Repository::findDatasetFile(kind, "ebike_riders.jpg");
    ASSERT_FALSE(demo.isEmpty());
    input->setText("user-selected-image.png");
    // Close a pending cold request, then deliver a late completion.
    ASSERT_TRUE(QFile::remove(archive));
    ASSERT_TRUE(QDir(Repository::extractPath(kind)).removeRecursively());
    auto* manager = repository.findChild<QNetworkAccessManager*>();
    ASSERT_TRUE(manager);
    manager->setProxy(QNetworkProxy(QNetworkProxy::HttpProxy, "127.0.0.1", 9));
    sample->click();
    ASSERT_TRUE(!sample->isEnabled());
    dialog.reject();
    ASSERT_TRUE(sample->isEnabled());
    ASSERT_TRUE(QDir().mkpath(QFileInfo(demo).absolutePath()));
    QFile lateDemo(demo);
    ASSERT_TRUE(lateDemo.open(QIODevice::WriteOnly));
    ASSERT_TRUE(lateDemo.write("P6\n1 1\n255\n\xff\0\0", 14) == 14);
    lateDemo.close();
    emit repository.datasetReady(true, kind);
    ASSERT_TRUE(input->text() == "user-selected-image.png");
    repository.cancelDownload();
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
