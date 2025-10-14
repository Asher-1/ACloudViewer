// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PYTHON_PLUGIN_PYTHON_ACTION_LAUNCHER_H
#define PYTHON_PLUGIN_PYTHON_ACTION_LAUNCHER_H

#include <QListWidget>
#include <QWidget>

class Ui_ActionLauncher;
class PythonPluginManager;
class PythonInterpreter;
class PluginListModel;

class PythonActionLauncher : public QWidget
{
    Q_OBJECT

    friend PluginListModel;

  public:
    explicit PythonActionLauncher(const PythonPluginManager *pluginManager,
                                  PythonInterpreter *interpreter,
                                  QWidget *parent = nullptr);

  protected:
    void showEvent(QShowEvent *event) override;
    void populateToolBox();
    void clearToolBox();
    void disable();
    void enable();

  private: // Members
    Ui_ActionLauncher *m_ui;
    const PythonPluginManager *m_pluginManager;
    PythonInterpreter *m_interpreter;
};

#endif // PYTHON_PLUGIN_PYTHON_ACTION_LAUNCHER_H
