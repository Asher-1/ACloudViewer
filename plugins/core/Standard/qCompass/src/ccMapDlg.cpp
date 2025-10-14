// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccMapDlg.h"

// ECV_DB_LIB
#include <CVLog.h>

// Qt
#include <qaction.h>
#include <qmenu.h>

#include <QApplication>
#include <QEvent>
#include <QKeyEvent>

// system
#include <assert.h>

ccMapDlg::ccMapDlg(QWidget* parent /*=0*/)
    : ccOverlayDialog(parent), Ui::mapDlg() {
    setupUi(this);

    // set background color
    QPalette p;
    p.setColor(backgroundRole(), QColor(240, 240, 240, 200));
    setPalette(p);
    setAutoFillBackground(true);

    // create menus
    m_createObject_menu = new QMenu(this);
    addObjectButton->setMenu(m_createObject_menu);

    // create actions
    m_create_geoObject = new QAction("GeoObject", this);
    m_create_geoObjectSS = new QAction("Single Surface GeoObject", this);

    // assign tool tips
    m_create_geoObject->setToolTip(
            "Create a GeoObject with upper and lower surfaces and an "
            "interior.");
    m_create_geoObjectSS->setToolTip(
            "Create a GeoObject with only a single surface ('interior').");

    // add to menu
    m_createObject_menu->addAction(m_create_geoObject);
    m_createObject_menu->addAction(m_create_geoObjectSS);
}
