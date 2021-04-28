//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#pragma once

//Qt
#include <QDialog>

class QSimpleUpdater;

namespace Ui {
class UpdateDialog;
}

class ecvUpdateDlg : public QDialog
{
    Q_OBJECT

public:
    explicit ecvUpdateDlg (QWidget* parent = nullptr);
    ~ecvUpdateDlg();

public slots:
    void resetFields();
    void checkForUpdates();
    void updateChangelog (const QString& url);
    void displayAppcast (const QString& url, const QByteArray& reply);

private:
    Ui::UpdateDialog* m_ui;
    QSimpleUpdater* m_updater;
};
