#pragma once

// ##########################################################################
// #                                                                        #
// #                              ACloudViewer                           #
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
// #                    COPYRIGHT: ACloudViewer project                  #
// #                                                                        #
// ##########################################################################

// qCC_db
#include <ecvHObject.h>

// Qt
#include <QDialog>

class Ui_EntitySelectionDialog;

//! Dialog to select one or multiple entities
class ecvEntitySelectionDialog : public QDialog {
    Q_OBJECT

public:
    //! Default constructor
    ecvEntitySelectionDialog(const ccHObject::Container& entities,
                             bool multiSelectionEnabled,
                             int defaultSelectedIndex = 0,
                             QWidget* parent = 0,
                             QString label = QString());

    //! Destructor
    virtual ~ecvEntitySelectionDialog();

    //! Returns the selected index (unique selection mode)
    int getSelectedIndex() const;

    //! Returns the selected indexes (multi-selection mode)
    void getSelectedIndexes(std::vector<int>& indexes) const;

    //! Static shortcut: unique selection mode
    static int SelectEntity(const ccHObject::Container& entities,
                            int defaultSelectedIndex = 0,
                            QWidget* parent = 0,
                            QString label = QString());

    //! Static shortcut: multi-selection mode
    static bool SelectEntities(const ccHObject::Container& entities,
                               std::vector<int>& indexes,
                               QWidget* parent = 0,
                               QString label = QString());

public slots:

    //! Selects all entities
    void selectAll();
    //! Selects all entities
    void selectNone();

protected:
    //! Associated ui
    Ui_EntitySelectionDialog* m_ui;
};
