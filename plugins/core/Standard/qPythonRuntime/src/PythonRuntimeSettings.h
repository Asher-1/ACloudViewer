// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <PythonConfig.h>

#include <QDialog>

class Ui_PythonRuntimeSettings;

class PythonRuntimeSettings final : public QDialog
{
  public:
    explicit PythonRuntimeSettings(QWidget *parent = nullptr);

    QStringList pluginsPaths() const;
    PythonConfig pythonEnvConfig() const;
    bool isDefaultPythonEnv() const;

  private: // Methods
    QString selectedEnvType() const;
    QString localEnvPath() const;

    void restoreSettings();
    void saveSettings() const;
    void handleEditPluginsPaths();
    void handleEnvComboBoxChange(const QString &envTypeName);
    void handleSelectLocalEnv();

  private:
    Ui_PythonRuntimeSettings *m_ui;
    QStringList m_pluginsPaths;
};
