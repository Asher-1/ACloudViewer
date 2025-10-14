// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PYTHON_PLUGIN_FILE_RUNNER_H
#define PYTHON_PLUGIN_FILE_RUNNER_H

#include <QDialog>

namespace Ui
{
class FileRunner;
}

class WaitingSpinnerWidget;
class PythonInterpreter;

/// The File Runner is a small widget that let the user pick a .py file
/// and run it, without having the editor opened.
class FileRunner final : public QDialog
{
    Q_OBJECT

  public:
    explicit FileRunner(PythonInterpreter *interp, QWidget *parent = nullptr);

    FileRunner(const FileRunner &) = delete;
    FileRunner operator=(const FileRunner &) = delete;
    FileRunner(FileRunner &&) = delete;
    FileRunner operator=(FileRunner &&) = delete;
    ~FileRunner() noexcept override;

  private Q_SLOTS:
    void selectFile();
    void runFile() const;
    void pythonExecutionStarted();
    void pythonExecutionEnded();

  protected:
    void resizeEvent(QResizeEvent *event) override;

  private:
    PythonInterpreter *m_interpreter;
    WaitingSpinnerWidget *m_busyWidget;
    Ui::FileRunner *m_ui;
    QString m_filePath;
};

#endif // PYTHON_PLUGIN_FILE_RUNNER_H
