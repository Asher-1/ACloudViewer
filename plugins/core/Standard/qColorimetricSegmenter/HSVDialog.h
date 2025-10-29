// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ##########################################################################
// #                                                                        #
// #            CLOUDVIEWER PLUGIN: ColorimetricSegmenter                   #
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
// #    COPYRIGHT:	Tri-Thien TRUONG, Ronan COLLIER, Mathieu LETRONE       #
// #                                                                        #
// ##########################################################################

#include <ecvPickingListener.h>
#include <ui_HSVDialog.h>

// Qt
#include <QDialog>

class ccPickingHub;

/*
        Get the values of the HSV interface, and interactions
*/
class HSVDialog : public QDialog,
                  public ccPickingListener,
                  public Ui::HSVDialog {
    Q_OBJECT
public:
    explicit HSVDialog(ccPickingHub* pickingHub, QWidget* parent = nullptr);

    //! Inherited from ccPickingListener
    virtual void onItemPicked(const PickedItem& pi);

public slots:
    void pickPoint(bool);
    void updateValues();

protected:  // members
    //! Picking hub
    ccPickingHub* m_pickingHub;
};
