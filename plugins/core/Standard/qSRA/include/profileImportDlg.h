// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QSRA_PROFILE_IMPORT_DLG_HEADER
#define QSRA_PROFILE_IMPORT_DLG_HEADER

#include "ui_profileImportDlg.h"

//! Dialog for importing a 2D revolution profile (qSRA plugin)
class ProfileImportDlg : public QDialog, public Ui::ProfileImportDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit ProfileImportDlg(QWidget* parent = 0);

    //! Returns revolution axis dimension index
    /** \return 0(X), 1(Y) or 2(Z).
     **/
    int getAxisDimension() const;

    //! Sets default filename
    void setDefaultFilename(QString filename);

    //! Returns input filename (on completion)
    QString getFilename() const;

    //! Returns whether the profile heights are absolute or not (i.e. relative
    //! to the center)
    bool absoluteHeightValues() const;

protected slots:

    //! Called when the 'browse' tool button is pressed
    void browseFile();
};

#endif  // QSRA_PROFILE_IMPORT_DLG_HEADER
