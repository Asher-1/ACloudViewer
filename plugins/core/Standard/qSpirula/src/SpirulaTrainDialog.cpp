// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "SpirulaTrainDialog.h"

#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProcess>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

#include "SpirulaLocator.h"

SpirulaTrainDialog::SpirulaTrainDialog(QWidget* parent) : QDialog(parent) {
    setWindowTitle(tr("Spirula Studio — Headless Training"));
    setModal(false);
    resize(640, 420);

    auto* layout = new QVBoxLayout(this);

    layout->addWidget(new QLabel(
            tr("Dataset folder (COLMAP model, Nerfstudio transforms.json, or "
               "plain images for the built-in SfM):"),
            this));
    m_datasetEdit = new QLineEdit(this);
    m_datasetEdit->setPlaceholderText(tr("/path/to/dataset"));
    layout->addWidget(m_datasetEdit);

    auto* row1 = new QHBoxLayout;
    row1->addWidget(new QLabel(tr("Run name:"), this));
    m_outputNameEdit = new QLineEdit(QStringLiteral("my_run"), this);
    row1->addWidget(m_outputNameEdit);
    row1->addWidget(new QLabel(tr("Device:"), this));
    m_deviceEdit = new QLineEdit(QStringLiteral("auto"), this);
    m_deviceEdit->setToolTip(
            tr(" spirula --device selector: auto, an index, a name substring, "
               "or uuid:hex"));
    row1->addWidget(m_deviceEdit);
    layout->addLayout(row1);

    layout->addWidget(new QLabel(
            tr("Extra arguments (optional, passed through verbatim):"), this));
    m_extraArgsEdit = new QLineEdit(this);
    m_extraArgsEdit->setPlaceholderText(
            QStringLiteral("--iters 30000 --enable-sfm …"));
    layout->addWidget(m_extraArgsEdit);

    auto* row2 = new QHBoxLayout;
    m_startButton = new QPushButton(tr("Start Training"), this);
    m_cancelButton = new QPushButton(tr("Cancel Training"), this);
    m_cancelButton->setEnabled(false);
    row2->addWidget(m_startButton);
    row2->addWidget(m_cancelButton);
    row2->addStretch();
    layout->addLayout(row2);

    m_logEdit = new QPlainTextEdit(this);
    m_logEdit->setReadOnly(true);
    layout->addWidget(m_logEdit);

    connect(m_startButton, &QPushButton::clicked, this,
            &SpirulaTrainDialog::startTraining);
    connect(m_cancelButton, &QPushButton::clicked, this,
            &SpirulaTrainDialog::cancelTraining);

    m_process = new QProcess(this);
    connect(m_process, &QProcess::readyReadStandardOutput, this,
            &SpirulaTrainDialog::onReadyReadStandardOutput);
    connect(m_process, &QProcess::readyReadStandardError, this,
            &SpirulaTrainDialog::onReadyReadStandardError);
    // finished is overloaded (int) / (int, ExitStatus) — pick the two-arg form
    connect(m_process,
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished), this,
            &SpirulaTrainDialog::onFinished);
}

SpirulaTrainDialog::~SpirulaTrainDialog() {
    // A detached-style long task must not be dragged down by dialog teardown:
    // the child keeps training and can be imported later via the run
    // directory.
    if (m_process && m_process->state() == QProcess::Running) {
        m_process->disconnect(this);
        m_process->terminate();
    }
}

void SpirulaTrainDialog::refreshExecutable() {
    m_exe = spirula::locateExecutable();
}

void SpirulaTrainDialog::setControlsRunning(bool running) {
    m_startButton->setEnabled(!running);
    m_cancelButton->setEnabled(running);
    m_datasetEdit->setEnabled(!running);
    m_outputNameEdit->setEnabled(!running);
    m_deviceEdit->setEnabled(!running);
    m_extraArgsEdit->setEnabled(!running);
}

void SpirulaTrainDialog::startTraining() {
    if (m_process->state() == QProcess::Running) {
        return;
    }
    const QString dataset = m_datasetEdit->text().trimmed();
    if (dataset.isEmpty() || !QFileInfo::exists(dataset)) {
        QMessageBox::warning(
                this, tr("Spirula Studio"),
                tr("Dataset folder does not exist:\n%1").arg(dataset));
        return;
    }
    if (m_exe.isEmpty()) {
        m_exe = spirula::locateExecutable();
    }
    if (m_exe.isEmpty()) {
        QMessageBox::warning(this, tr("Spirula Studio"),
                             tr("Spirula Studio executable not found. Build "
                                "it with -DBUILD_SPIRULA=ON."));
        return;
    }

    QStringList args{QStringLiteral("train"), dataset};
    const QString runName = m_outputNameEdit->text().trimmed();
    if (!runName.isEmpty()) {
        args << QStringLiteral("--output-dir-name") << runName;
    }
    const QString device = m_deviceEdit->text().trimmed();
    if (!device.isEmpty() && device != QStringLiteral("auto")) {
        args << QStringLiteral("--device") << device;
    }
    const QStringList extra =
            m_extraArgsEdit->text().split(' ', Qt::SkipEmptyParts);
    for (const QString& arg : extra) {
        args << arg;
    }

    m_logEdit->clear();
    m_logEdit->appendPlainText(m_exe + ' ' + args.join(' '));
    m_process->setProgram(m_exe);
    m_process->setArguments(args);
    m_process->setWorkingDirectory(QFileInfo(m_exe).absolutePath());
    m_process->start();
    setControlsRunning(true);
}

void SpirulaTrainDialog::cancelTraining() {
    if (m_process->state() != QProcess::Running) {
        return;
    }
    // Upstream handles SIGTERM as a graceful stop (checkpoint on exit).
    m_process->terminate();
    if (!m_process->waitForFinished(10000)) {
        m_process->kill();
    }
}

void SpirulaTrainDialog::onReadyReadStandardOutput() {
    m_process->setReadChannel(QProcess::StandardOutput);
    m_logEdit->appendPlainText(
            QString::fromUtf8(m_process->readAllStandardOutput()).trimmed());
    m_logEdit->verticalScrollBar()->setValue(
            m_logEdit->verticalScrollBar()->maximum());
}

void SpirulaTrainDialog::onReadyReadStandardError() {
    m_logEdit->appendPlainText(
            QString::fromUtf8(m_process->readAllStandardError()).trimmed());
    m_logEdit->verticalScrollBar()->setValue(
            m_logEdit->verticalScrollBar()->maximum());
}

void SpirulaTrainDialog::onFinished(int exitCode, QProcess::ExitStatus status) {
    setControlsRunning(false);
    if (status == QProcess::CrashExit) {
        QMessageBox::critical(this, tr("Spirula Studio"),
                              tr("Training crashed — see the log above and "
                                 "<config>/crash.log."));
        return;
    }
    if (exitCode != 0) {
        QMessageBox::warning(this, tr("Spirula Studio"),
                             tr("Training exited with code %1.").arg(exitCode));
        return;
    }

    // Upstream layout: <cwd>/outputs/<run name>/ with config.json, checkpoints
    // and splat.ply — resolveSplatPath accepts the run directory directly.
    const QString runName = m_outputNameEdit->text().trimmed();
    const QString runDir =
            QDir(QFileInfo(m_exe).absolutePath())
                    .filePath(QStringLiteral("outputs/%1")
                                      .arg(runName.isEmpty()
                                                   ? QStringLiteral("outputs")
                                                   : runName));
    m_logEdit->appendPlainText(tr("Training finished: %1").arg(runDir));

    const auto answer = QMessageBox::question(
            this, tr("Spirula Studio"),
            tr("Training finished. Import the result into the DB tree now?"),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
    if (answer == QMessageBox::Yes) {
        emit runFinished(runDir);
    }
}
