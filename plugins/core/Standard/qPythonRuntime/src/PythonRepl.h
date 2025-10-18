// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QMainWindow>

#include <string>

#include "PythonInterpreter.h"

class PythonInterpreter;
class Ui_PythonREPL;
class QPlainTextEdit;

class PythonRepl;

/// Simple history system for the REPL.
class History final
{
  public:
    explicit History() = default;

    void add(QString &&cmd);

    size_t size() const;

    bool empty() const;

    const QString &older();

    const QString &newer();

  private:
    QVector<QString> m_commands;
    QVector<QString>::const_reverse_iterator m_current;
};

/// Class used by the REPL to handle key presses
class KeyPressEater final : public QObject
{
    Q_OBJECT
  public:
    explicit KeyPressEater(PythonRepl *repl, QObject *parent = nullptr);

  protected:
    bool eventFilter(QObject *obj, QEvent *event) override;

  private:
    PythonRepl *m_repl{nullptr};
};

/// Homemade REPL (Read Print Eval Loop)
class PythonRepl final : public QMainWindow
{
    friend KeyPressEater;
    Q_OBJECT

  public:
    explicit PythonRepl(PythonInterpreter *interpreter, QMainWindow *parent = nullptr);

    void executeCode(const QString &pythonCode);

    PythonRepl(const PythonRepl &) = delete;
    PythonRepl(PythonRepl &&) = delete;
    PythonRepl &operator=(const PythonRepl &) = delete;
    PythonRepl &operator=(PythonRepl &&) = delete;
    ~PythonRepl() noexcept override;

  protected:
    QPlainTextEdit *codeEdit();
    QListWidget *outputDisplay();

    void reset();
    void importNeededPackages();
    void setupUI();

  private:
    History m_history{};
    std::string m_buf;
    Ui_PythonREPL *m_ui{nullptr};
    PythonInterpreter *m_interpreter{nullptr};
    PythonInterpreter::State m_state;
};
