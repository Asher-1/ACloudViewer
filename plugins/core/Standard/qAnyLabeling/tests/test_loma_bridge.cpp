// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QComboBox>
#include <QCryptographicHash>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPushButton>
#include <QTableWidget>
#include <QTimer>
#include <cmath>
#include <cstdio>

#include "LomaBridgeDialog.h"
#include "ecvTestDataRepository.h"

namespace {
QString inputPath(const char* variable, const QString& filename) {
    if (qEnvironmentVariableIsSet(variable))
        return QString::fromLocal8Bit(qgetenv(variable));
    return ecvTestDataRepository::findDatasetFile(
            ecvTestDataRepository::Dataset::LightGlueTestImages, filename);
}

bool verifyInput(const QString& path, const QString& id, bool custom) {
    QFile input(path);
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!input.open(QIODevice::ReadOnly) || !hash.addData(&input)) return false;
    const QByteArray digest = hash.result().toHex();
    std::printf("LoMa input: %s sha256=%s\n", qPrintable(path),
                digest.constData());
    if (custom) return true;
    QFile manifest(QStringLiteral(QANY_VALIDATION_MANIFEST));
    if (!manifest.open(QIODevice::ReadOnly)) return false;
    const auto assets = QJsonDocument::fromJson(manifest.readAll())
                                .object()
                                .value(QStringLiteral("input_assets"))
                                .toArray();
    for (const auto& value : assets) {
        const auto asset = value.toObject();
        if (asset.value(QStringLiteral("id")).toString() == id)
            return digest ==
                   asset.value(QStringLiteral("sha256")).toString().toLatin1();
    }
    return false;
}
}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    const QString imageA =
            inputPath("QANY_LOMA_IMAGE_A", QStringLiteral("sacre_coeur1.jpg"));
    const QString imageB =
            inputPath("QANY_LOMA_IMAGE_B", QStringLiteral("sacre_coeur2.jpg"));
    if (!verifyInput(imageA, QStringLiteral("sacre-coeur-1"),
                     qEnvironmentVariableIsSet("QANY_LOMA_IMAGE_A")) ||
        !verifyInput(imageB, QStringLiteral("sacre-coeur-2"),
                     qEnvironmentVariableIsSet("QANY_LOMA_IMAGE_B"))) {
        std::fprintf(
                stderr,
                "FAIL: missing or corrupt LoMa image pair. Prepare the pinned "
                "lightglue_test_images fixtures in the shared data cache, or "
                "set QANY_LOMA_IMAGE_A/B to a real matching pair.\n");
        return 1;
    }
    const bool lightglue = qEnvironmentVariableIsSet("QANY_LIGHTGLUE_TEST");
    LomaBridgeDialog dialog(lightglue, nullptr);
    dialog.setImage0(imageA);
    dialog.setImage1(imageB);
    struct ModelChoice {
        const char* name;
        const char* file;
    };
    const ModelChoice choices[] = {
            {"LomaDetectorModel", "loma_detector.f16.gguf"},
            {"LomaDescriptorModel", "loma_descriptor_dedode_g.f16.gguf"},
            {"LomaMatcherModel", "loma_matcher_B.f16.gguf"}};
    const ModelChoice lightglueChoices[] = {
            {"LomaDetectorModel", "aliked-n16rot-f16.gguf"},
            {"LomaMatcherModel", "aliked-lightglue-f16.gguf"}};
    const QVector<ModelChoice> selectedChoices =
            lightglue
                    ? QVector<ModelChoice>{lightglueChoices[0],
                                           lightglueChoices[1]}
                    : QVector<ModelChoice>{choices[0], choices[1], choices[2]};
    for (const auto& choice : selectedChoices) {
        auto* combo =
                dialog.findChild<QComboBox*>(QString::fromLatin1(choice.name));
        if (!combo) return 1;
        int selected = -1;
        for (int i = 0; i < combo->count(); ++i) {
            if (QFileInfo(combo->itemData(i).toString()).fileName() ==
                QString::fromLatin1(choice.file)) {
                selected = i;
                break;
            }
        }
        if (selected < 0) return 1;
        if (!QFileInfo(combo->itemData(selected).toString()).isFile()) {
            std::fprintf(stderr, "FAIL: missing catalog model %s\n",
                         choice.file);
            return 1;
        }
        combo->setCurrentIndex(selected);
    }
    auto* backend = dialog.findChild<QComboBox*>(QStringLiteral("LomaBackend"));
    const QByteArray selectedBackend = qgetenv("QANY_LOMA_BACKEND");
    if (!backend) return 1;
    backend->setCurrentText(selectedBackend.isEmpty()
                                    ? QStringLiteral("cpu")
                                    : QString::fromLocal8Bit(selectedBackend));
    QEventLoop loop;
    bool finished = false;
    bool success = false;
    int matchCount = 0;
    QObject::connect(&dialog, &LomaBridgeDialog::matchingFinished, &loop,
                     [&](bool ok, int count) {
                         finished = true;
                         success = ok;
                         matchCount = count;
                         loop.quit();
                     });
    QTimer timer;
    timer.setSingleShot(true);
    QObject::connect(&timer, &QTimer::timeout, &loop, &QEventLoop::quit);
    timer.start(300000);
    QTimer::singleShot(0, &dialog, [&dialog]() {
        QMetaObject::invokeMethod(&dialog, "runMatching", Qt::DirectConnection);
    });
    loop.exec();
    if (!finished || !success || matchCount <= 0) {
        std::fprintf(stderr,
                     "LoMa bridge failed: finished=%d success=%d matches=%d\n",
                     finished, success, matchCount);
        return 1;
    }
    const auto accepted = dialog.acceptedMatches();
    if (accepted.size() != matchCount) return 1;
    const QSize sizeA = QImageReader(imageA).size(),
                sizeB = QImageReader(imageB).size();
    for (const auto& match : accepted) {
        if (!std::isfinite(match.score) || match.score < 0.1 ||
            match.score > 1.0 || !std::isfinite(match.image0.x()) ||
            !std::isfinite(match.image0.y()) ||
            !std::isfinite(match.image1.x()) ||
            !std::isfinite(match.image1.y()) || match.image0.x() < 0 ||
            match.image0.y() < 0 || match.image1.x() < 0 ||
            match.image1.y() < 0 || match.image0.x() >= sizeA.width() ||
            match.image0.y() >= sizeA.height() ||
            match.image1.x() >= sizeB.width() ||
            match.image1.y() >= sizeB.height())
            return 1;
    }
    auto* table = dialog.findChild<QTableWidget*>();
    if (!table || table->rowCount() != matchCount) return 1;
    table->item(0, 0)->setCheckState(Qt::Unchecked);
    if (dialog.acceptedMatches().size() != matchCount - 1) return 1;
    std::printf("%s real pair matches: %d\n", lightglue ? "LightGlue" : "LoMa",
                matchCount);
    dialog.setImage1(imageB);
    if (!dialog.acceptedMatches().isEmpty() || table->rowCount() != 0) {
        std::fprintf(stderr, "Changing an image retained stale LoMa matches\n");
        return 1;
    }
    // Test the delivery guard using the real run: invalidate the input
    // after dispatch and verify that its queued completion cannot be accepted.
    if (qEnvironmentVariableIsSet("QANY_LOMA_TEST_STALE")) {
        finished = false;
        success = true;
        QTimer::singleShot(0, &dialog, [&]() {
            QMetaObject::invokeMethod(&dialog, "runMatching",
                                      Qt::DirectConnection);
            dialog.setImage0(imageA);
        });
        timer.start(300000);
        loop.exec();
        if (!finished || success || !dialog.acceptedMatches().isEmpty()) {
            std::fprintf(stderr,
                         "LoMa accepted a result for replaced inputs\n");
            return 1;
        }
        std::puts("LoMa stale completion rejected: PASS");
    }
    return 0;
}
