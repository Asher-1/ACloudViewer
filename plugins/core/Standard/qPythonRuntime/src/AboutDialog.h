// ##########################################################################
// #                                                                        #
// #                ACLOUDVIEWER PLUGIN: PythonRuntime                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 of the License.               #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                   COPYRIGHT: Thomas Montaigu                           #
// #                                                                        #
// ##########################################################################

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
