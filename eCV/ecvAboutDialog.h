// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECVABOUTDIALOG_H
#define ECVABOUTDIALOG_H

// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: CLOUDVIEWER  project                               #
// #                                                                        #
// ##########################################################################

#include <QDialog>

namespace Ui {
class AboutDialog;
}

class ecvAboutDialog : public QDialog {
    Q_OBJECT

public:
    ecvAboutDialog(QWidget *parent = nullptr);
    ~ecvAboutDialog();

private:
    Ui::AboutDialog *mUI;
};

#endif  // ECVABOUTDIALOG_H
