// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QComboBox>
#include <QEventLoop>
#include <QFileInfo>
#include <QTimer>
#include <cstdio>

#include "LomaBridgeDialog.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    const QString imageA = QString::fromLocal8Bit(qgetenv("QANY_LOMA_IMAGE_A"));
    const QString imageB = QString::fromLocal8Bit(qgetenv("QANY_LOMA_IMAGE_B"));
    if (!QFileInfo(imageA).isFile() || !QFileInfo(imageB).isFile()) {
        std::fprintf(stderr,
                     "SKIP: set QANY_LOMA_IMAGE_A and QANY_LOMA_IMAGE_B\n");
        return 77;
    }
    LomaBridgeDialog dialog;
    dialog.setImage0(imageA);
    dialog.setImage1(imageB);
    struct ModelChoice {
        const char* name;
        const char* file;
    };
    const ModelChoice choices[] = {
            {"LomaDetectorModel", "loma_detector.f32.gguf"},
            {"LomaDescriptorModel", "loma_descriptor_dedode_g.f32.gguf"},
            {"LomaMatcherModel", "loma_matcher_B.f32.gguf"}};
    for (const auto& choice : choices) {
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
    for (const auto& match : dialog.acceptedMatches()) {
        if (match.image0.x() < 0 || match.image0.y() < 0 ||
            match.image1.x() < 0 || match.image1.y() < 0)
            return 1;
    }
    std::printf("LoMa real pair matches: %d\n", matchCount);
    return 0;
}
