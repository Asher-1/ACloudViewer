// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_MAP_DIALOG_HEADER
#define ECV_MAP_DIALOG_HEADER

// Qt
#include <ecvOverlayDialog.h>

#include <QAction>
#include <QDialog>
#include <QList>

// Local
#include <ui_mapDlg.h>

#include "ccTrace.h"

// class encapsulating the map-mode overlay dialog
class ccMapDlg : public ccOverlayDialog, public Ui::mapDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccMapDlg(QWidget *parent = 0);

    // menus
    QMenu *m_createObject_menu;

    // actions
    QAction *m_create_geoObject;    // create a normal GeoObject
    QAction *m_create_geoObjectSS;  // create a single surface GeoObject
};

#endif  // ECV_MAP_DIALOG_HEADER
