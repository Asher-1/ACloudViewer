// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Focused qOCR workspace contract test: bilingual model selection, the
// device/threads runtime row, the shared progress section, the DB-export
// switch, the unified download entry, one-click official test data, and the
// shared-console logging contract (no private log widget).

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QTableWidget>
#include <cstdio>

#include "OCRDialog.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    // Isolate from the developer machine's persisted qOCR state (a staged
    // image from a GUI session would enable Run before the test stages it).
    {
        QSettings settings;
        settings.beginGroup(QStringLiteral("qOCR"));
        settings.remove(QStringLiteral(""));
        settings.endGroup();
    }

    OCRDialog dialog;
    auto* language =
            dialog.findChild<QComboBox*>(QStringLiteral("qOCRLanguage"));
    auto* quant = dialog.findChild<QComboBox*>(QStringLiteral("qOCRQuant"));
    auto* device = dialog.findChild<QComboBox*>(QStringLiteral("qOCRDevice"));
    auto* threads = dialog.findChild<QSpinBox*>(QStringLiteral("qOCRThreads"));
    auto* imagePath =
            dialog.findChild<QLineEdit*>(QStringLiteral("qOCRImagePath"));
    auto* testData =
            dialog.findChild<QPushButton*>(QStringLiteral("qOCRUseTestData"));
    auto* download =
            dialog.findChild<QPushButton*>(QStringLiteral("qOCRDownload"));
    auto* run = dialog.findChild<QPushButton*>(QStringLiteral("qOCRRun"));
    auto* cancel = dialog.findChild<QPushButton*>(QStringLiteral("qOCRCancel"));
    auto* addDb = dialog.findChild<QCheckBox*>(QStringLiteral("qOCRAddToDb"));
    auto* results =
            dialog.findChild<QTableWidget*>(QStringLiteral("qOCRResults"));
    auto* progress =
            dialog.findChild<QProgressBar*>(QStringLiteral("qOCRProgress"));
    if (!language || !quant || !device || !threads || !imagePath || !testData ||
        !download || !run || !cancel || !addDb || !results || !progress) {
        std::fprintf(stderr, "qOCR workspace is missing a core widget\n");
        return 1;
    }

    // Shared-console logging contract: the private log widget is gone; all
    // messages route through aicore_inference_log -> the application console.
    if (dialog.findChild<QPlainTextEdit*>(QStringLiteral("qOCRLog"))) {
        std::fprintf(stderr,
                     "qOCR must not embed a private log widget (use the "
                     "shared console)\n");
        return 1;
    }

    // Bilingual catalog: Chinese and English rec models over the shared
    // script-agnostic detector.
    if (language->count() != 2 ||
        language->itemData(0).toString() != QStringLiteral("ch") ||
        language->itemData(1).toString() != QStringLiteral("en")) {
        std::fprintf(stderr, "qOCR language selection must offer ch and en\n");
        return 1;
    }
    if (quant->count() != 3) {
        std::fprintf(stderr, "qOCR must expose the F32/F16/Q8_0 tiers\n");
        return 1;
    }
    if (results->columnCount() != 4) {
        std::fprintf(stderr, "qOCR results table must show 4 columns\n");
        return 1;
    }

    // Device selection follows the shared runtime row: at least the
    // synthetic "auto" entry, plus a threads spinner defaulting to auto.
    if (device->count() < 1 || device->itemData(0).toString().isEmpty()) {
        std::fprintf(stderr, "qOCR device combo must offer the auto entry\n");
        return 1;
    }
    if (threads->value() != 0) {
        std::fprintf(stderr, "qOCR threads must default to 0 (= auto)\n");
        return 1;
    }

    // The run starts disabled until models and an image are staged; the
    // cancel button mirrors the running state.
    if (run->isEnabled()) {
        std::fprintf(stderr,
                     "qOCR Run must start disabled until models and an "
                     "image are staged\n");
        return 1;
    }
    if (cancel->isEnabled()) {
        std::fprintf(stderr, "qOCR Cancel must start disabled\n");
        return 1;
    }

    // Switching the language retargets the rec model and keeps the state
    // line informative (the det file is shared and language-agnostic).
    language->setCurrentIndex(1);
    auto* status = dialog.findChild<QLabel*>(QStringLiteral("qOCRStatus"));
    if (!status || !status->text().contains(QStringLiteral("det:")) ||
        !status->text().contains(QStringLiteral("rec:"))) {
        std::fprintf(stderr, "qOCR status line missing model state\n");
        return 1;
    }

    // Filling a test image is a pure UI action when the asset is already
    // cached; verify the path editor accepts and reflects it.
    imagePath->setText(QStringLiteral("/tmp/qocr-fixture.png"));
    if (imagePath->text() != QStringLiteral("/tmp/qocr-fixture.png")) {
        std::fprintf(stderr, "qOCR image path editor failed to update\n");
        return 1;
    }

    std::printf("qOCR workspace contract OK\n");
    return 0;
}
