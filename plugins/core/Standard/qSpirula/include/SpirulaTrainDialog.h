// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QProcess>

class QLineEdit;
class QPlainTextEdit;
class QPushButton;

//! Headless training wizard: drives `spirula train <dataset>` as a detached
//! child process with live stdout, cancel support, and a one-click hand-off
//! of the finished run directory to the DB-tree importer.
//!
//! Output parsing is deliberately format-agnostic (raw stdout tail + exit
//! code) so upstream log changes ride the SPIRULA_TAG upgrade for free.
class SpirulaTrainDialog : public QDialog {
    Q_OBJECT

public:
    explicit SpirulaTrainDialog(QWidget* parent = nullptr);
    ~SpirulaTrainDialog() override;

    //! Refreshes the executable line (called before show()).
    void refreshExecutable();

signals:
    //! Emitted when a training run finished successfully; carries the run
    //! directory (resolvable by SplatImporter/resolveSplatPath).
    void runFinished(const QString& runDir);

private slots:
    void startTraining();
    void cancelTraining();
    void onReadyReadStandardOutput();
    void onReadyReadStandardError();
    void onFinished(int exitCode, QProcess::ExitStatus status);

private:
    void setControlsRunning(bool running);

    QString m_exe;
    QProcess* m_process = nullptr;
    QLineEdit* m_datasetEdit = nullptr;
    QLineEdit* m_outputNameEdit = nullptr;
    QLineEdit* m_deviceEdit = nullptr;
    QLineEdit* m_extraArgsEdit = nullptr;
    QPushButton* m_startButton = nullptr;
    QPushButton* m_cancelButton = nullptr;
    QPlainTextEdit* m_logEdit = nullptr;
};
