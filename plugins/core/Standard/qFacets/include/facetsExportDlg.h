// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

#include "ui_facetsExportDlg.h"

//! Dialog for exporting facets or facets info (qFacets plugin)
class FacetsExportDlg : public QDialog, public Ui::FacetsExportDlg {
    Q_OBJECT

public:
    //! Usage mode
    enum IOMode { SHAPE_FILE_IO, ASCII_FILE_IO };

    //! Default constructor
    FacetsExportDlg(IOMode mode, QWidget* parent = 0);

protected slots:

    //! Called when the 'browse' tool button is pressed
    void browseDestination();

protected:
    //! Current I/O mode
    IOMode m_mode;
};
