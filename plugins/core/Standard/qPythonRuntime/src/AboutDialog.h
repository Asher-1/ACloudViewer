// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

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
