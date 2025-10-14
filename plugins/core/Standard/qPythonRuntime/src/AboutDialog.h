// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PYTHON_PLUGIN_PROJECTS_ABOUT_DIALOG_H
#define PYTHON_PLUGIN_PROJECTS_ABOUT_DIALOG_H

#include <QDialog>

class Ui_AboutDialog;

class AboutDialog final : public QDialog
{
  public:
    explicit AboutDialog(QWidget *parent = nullptr);

    ~AboutDialog() noexcept override;

  private:
    Ui_AboutDialog *m_dlg;
};

#endif // PYTHON_PLUGIN_PROJECTS_ABOUT_DIALOG_H
