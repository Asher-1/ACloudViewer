// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PYTHON_PLUGIN_PACKAGE_MANAGER_H
#define PYTHON_PLUGIN_PACKAGE_MANAGER_H

#include <ui_PackageManager.h>

class QProcess;
class CommandOutputDialog;
class PythonConfig;

/// Widget that shows an interface allowing the user
/// to see the list of currently installed packages in the
/// current environment.
/// It also allows to install / uninstall packages.
///
/// It works by wrapping pip using QProcess.
class PackageManager final : public QWidget
{
    Q_OBJECT
  public:
    explicit PackageManager(const PythonConfig &config, QWidget *parent = nullptr);
    ~PackageManager() noexcept override;

  private: // Methods
    void refreshInstalledPackagesList();
    void handleInstallPackage();
    void handleUninstallPackage();
    void handleSelectionChanged();
    void handleSearch();

    void executeCommand(const QStringList &arguments);
    void setBusy(bool isBusy);

  private: // Members
    Ui_PackageManager *m_ui;
    QProcess *m_pythonProcess;
    CommandOutputDialog *m_outputDialog;
    bool m_shouldUseUserOption;
};

#endif // PYTHON_PLUGIN_PACKAGE_MANAGER_H
