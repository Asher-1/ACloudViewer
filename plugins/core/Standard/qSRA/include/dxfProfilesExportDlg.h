// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ui_dxfProfilesExportDlg.h"

//! Dialog for export multiple 2D profiles in a single DXF file (qSRA plugin)
class DxfProfilesExportDlg : public QDialog, public Ui::DxfProfilesExportDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit DxfProfilesExportDlg(QWidget* parent = 0);

    //! Returns vert. profiles output filename (on completion)
    QString getVertFilename() const;
    //! Returns horiz. profiles output filename (on completion)
    QString getHorizFilename() const;

protected slots:

    //! Called when the vert. 'browse' tool button is pressed
    void browseVertFile();
    //! Called when the horiz. 'browse' tool button is pressed
    void browseHorizFile();

    //! Saves dialog state to persistent settings
    void acceptAndSaveSettings();

protected:
    //! Inits dialog state from persistent settings
    void initFromPersistentSettings();
};
