// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

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

#include <QObject>

class QAction;
class QMenu;
class QString;
class QStringList;

class ecvRecentFiles : public QObject {
    Q_OBJECT

public:
    ecvRecentFiles(QWidget *parent);

    //! Returns a "most recently used file" menu
    QMenu *menu();

    //! Adds a file path to the recently used menu
    void addFilePath(const QString &filePath);

private:
    //! Updates the contents of the menu
    void updateMenu();

    //! Opens a file based on the action that was triggered
    void openFileFromAction();

    //! Returns a list of file paths from the QSettings
    //! This will also remove any file from the list that does not exist
    QStringList listRecent();

    //! Contracts the path by substituting '~' for the user's home directory
    QString contractFilePath(const QString &filePath);

    //! Expands the path by substituting the user's home directory for '~'
    QString expandFilePath(const QString &filePath);

    //! The key in the QSettings where we store the file list
    static QString s_settingKey;

    QMenu *m_menu;

    QAction *m_actionClearMenu;
};
