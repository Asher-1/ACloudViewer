// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "facetsExportDlg.h"

// Qt
#include <QFileDialog>

// System
#include <assert.h>

FacetsExportDlg::FacetsExportDlg(IOMode mode, QWidget* parent)
    : QDialog(parent, Qt::Tool), Ui::FacetsExportDlg(), m_mode(mode) {
    setupUi(this);

    connect(browseToolButton, &QAbstractButton::clicked, this,
            &FacetsExportDlg::browseDestination);
}

void FacetsExportDlg::browseDestination() {
    QString saveFileFilter;
    switch (m_mode) {
        case SHAPE_FILE_IO:
            saveFileFilter = "Shapefile (*.shp)";
            break;
        case ASCII_FILE_IO:
            saveFileFilter = "ASCII table (*.csv)";
            break;
        default:
            assert(false);
            return;
    }

    // open file saving dialog
    QString outputFilename = QFileDialog::getSaveFileName(
            0, "Select destination", destinationPathLineEdit->text(),
            saveFileFilter);

    if (outputFilename.isEmpty()) return;

    destinationPathLineEdit->setText(outputFilename);
}
