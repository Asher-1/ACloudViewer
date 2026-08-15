// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVLog.h>
#include <ecvTestDataRepository.h>

#include <QDir>
#include <QObject>
#include <QProgressDialog>
#include <QSet>
#include <QString>
#include <QWidget>
#include <functional>

namespace qmcalib_ui {

/**
 * @brief Provision the qManualCalib sample test data with a live progress bar.
 *
 * Drives the shared ecvTestDataRepository singleton through the full
 * download -> extract -> apply pipeline:
 *   - If the dataset is already extracted, invokes onReady(true) immediately.
 *   - Otherwise shows a modal QProgressDialog and streams download / extraction
 *     progress via the repository's existing signal callbacks.
 *   - Calls onReady(success) when provisioning completes (success=false on any
 *     download/extraction failure).
 *
 * Stability guarantees:
 *   - Re-entrancy is guarded: clicking again while a provisioning is active
 *     for the same dialog is ignored, so duplicate connections cannot pile up.
 *   - Every repo -> parent connection created here is torn down in `finish`,
 *     so no stale slot can later fire against a dead progress dialog.
 *   - `finish` is guaranteed to run exactly once per provisioning.
 */
inline void runManualCalibTestDataProvision(
        QWidget* parent, const std::function<void(bool)>& onReady) {
    using TestDataset = ecvTestDataRepository::Dataset;
    auto& repo = ecvTestDataRepository::instance();
    const TestDataset kind = TestDataset::ManualCalib;

    // Re-entrancy guard: ignore clicks while a provisioning is active for this
    // dialog. Prevents duplicate connections and double-deletion crashes.
    static QSet<QWidget*> s_active;
    if (!parent || s_active.contains(parent)) {
        return;
    }
    s_active.insert(parent);

    // 1. Already extracted -> nothing to do.
    if (QDir(ecvTestDataRepository::extractPath(kind)).exists()) {
        s_active.remove(parent);
        if (onReady) onReady(true);
        return;
    }

    auto* progress = new QProgressDialog(QObject::tr("Preparing test data..."),
                                         QObject::tr("Cancel"), 0, 100, parent);
    progress->setWindowModality(Qt::WindowModal);
    progress->setMinimumDuration(0);
    progress->setValue(0);
    progress->setAutoClose(false);
    progress->setAutoReset(false);

    // Cancel stops any in-progress download; the failure signal then closes
    // the dialog deterministically.
    QObject::connect(progress, &QProgressDialog::canceled, &repo,
                     &ecvTestDataRepository::cancelDownload,
                     Qt::UniqueConnection);

    // Single tear-down: close the dialog, remove every connection this helper
    // installed on `parent`, release the re-entrancy guard, then hand control
    // back to the caller.
    const auto finish = [&repo, parent, progress, onReady](bool ok) {
        progress->close();
        progress->deleteLater();
        QObject::disconnect(&repo, &ecvTestDataRepository::downloadProgress,
                            parent, nullptr);
        QObject::disconnect(&repo, &ecvTestDataRepository::downloadLogMessage,
                            parent, nullptr);
        QObject::disconnect(&repo, &ecvTestDataRepository::downloadFinished,
                            parent, nullptr);
        QObject::disconnect(&repo, &ecvTestDataRepository::extractionProgress,
                            parent, nullptr);
        QObject::disconnect(&repo, &ecvTestDataRepository::extractionFinished,
                            parent, nullptr);
        s_active.remove(parent);
        if (onReady) onReady(ok);
    };

    // Download progress (0-100) + live status text.
    QObject::connect(
            &repo, &ecvTestDataRepository::downloadProgress, parent,
            [progress](int percent, const QString& statusText) {
                if (!statusText.isEmpty()) {
                    progress->setLabelText(statusText);
                }
                progress->setValue(percent);
            },
            Qt::UniqueConnection);

    // Surface repository log messages (info/errors) to the console so
    // failures are visible instead of silent.
    QObject::connect(
            &repo, &ecvTestDataRepository::downloadLogMessage, parent,
            [](const QString& msg) {
                CVLog::Print("%s", msg.toUtf8().constData());
            },
            Qt::UniqueConnection);

    // Extraction progress (current/total entries) mapped to 0-100.
    QObject::connect(
            &repo, &ecvTestDataRepository::extractionProgress, parent,
            [progress](int current, int total) {
                progress->setLabelText(QObject::tr("Extracting test data..."));
                progress->setValue(total > 0 ? (current * 100) / total : 0);
            },
            Qt::UniqueConnection);

    // Extraction completion is the single endpoint for both the cached-zip and
    // download paths. `extractDataset()` emits extractionFinished on every
    // outcome (success or failure).
    QObject::connect(
            &repo, &ecvTestDataRepository::extractionFinished, parent,
            [finish](bool ok, TestDataset) { finish(ok); },
            Qt::UniqueConnection);

    // A successful download triggers extraction; a failed one tears down.
    QObject::connect(
            &repo, &ecvTestDataRepository::downloadFinished, parent,
            [&repo, kind, finish](bool ok, TestDataset) {
                if (!ok) {
                    finish(false);
                    return;
                }
                repo.extractDataset(kind);
            },
            Qt::UniqueConnection);

    // 2. Zip already cached -> extract only.
    if (repo.isDatasetAvailable(kind)) {
        repo.extractDataset(kind);
        return;
    }

    // 3. Download -> (downloadFinished handler) extract -> finish.
    repo.startDownload(kind);
}

}  // namespace qmcalib_ui
