// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ui_FileRunner.h"

#include "FileRunner.h"
#include "PythonInterpreter.h"
#include "Resources.h"
#include "WaitingSpinnerWidget.h"

#include <QFileDialog>
#include <QResizeEvent>
#include <QStyle>

FileRunner::FileRunner(PythonInterpreter *interp, QWidget *parent)
    : QDialog(parent), m_interpreter(interp), m_busyWidget(nullptr), m_ui(new Ui::FileRunner)

{
    m_ui->setupUi(this);
    m_ui->runFileBtn->setEnabled(false);
    m_ui->runFileBtn->setIcon(QApplication::style()->standardIcon(QStyle::SP_ArrowRight));
    setWindowIcon(QIcon(RUNNER_ICON_PATH));
    m_busyWidget = new WaitingSpinnerWidget(this);

    connect(m_ui->selectFileBtn, &QPushButton::clicked, this, &FileRunner::selectFile);
    connect(m_ui->runFileBtn, &QPushButton::clicked, this, &FileRunner::runFile);
    connect(
        interp, &PythonInterpreter::executionStarted, this, &FileRunner::pythonExecutionStarted);
    connect(interp, &PythonInterpreter::executionFinished, this, &FileRunner::pythonExecutionEnded);
}

void FileRunner::selectFile()
{
    m_filePath = QFileDialog::getOpenFileName(this,
                                              QStringLiteral("Select Python Script"),
                                              QString(),
                                              QStringLiteral("Python Script (*.py)"));
    m_ui->filePathLabel->setText(m_filePath);
    m_ui->runFileBtn->setEnabled(!m_filePath.isEmpty());
}

void FileRunner::runFile() const
{
    if (!m_filePath.isEmpty() && m_interpreter)
    {
        const std::string path = m_filePath.toStdString();
        m_interpreter->executeFile(path);
    }
}

void FileRunner::pythonExecutionStarted()
{
    setEnabled(false);
    m_busyWidget->start();
}

void FileRunner::pythonExecutionEnded()
{
    setEnabled(true);
    m_busyWidget->stop();
}

FileRunner::~FileRunner() noexcept
{
    delete m_ui;
}

void FileRunner::resizeEvent(QResizeEvent *event)
{
    m_busyWidget->resize(event->size());
    QDialog::resizeEvent(event);
}
