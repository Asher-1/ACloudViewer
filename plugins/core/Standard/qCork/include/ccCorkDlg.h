// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_CORK_DLG_HEADER
#define ECV_CORK_DLG_HEADER

#include "ui_corkDlg.h"

//! Dialog for qCork plugin
class ccCorkDlg : public QDialog, public Ui::CorkDialog {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccCorkDlg(QWidget* parent = 0);

    //! Supported CSG operations
    enum CSG_OPERATION { UNION, INTERSECT, DIFF, SYM_DIFF };

    //! Set meshes names
    void setNames(QString A, QString B);

    //! Returns the selected operation
    CSG_OPERATION getSelectedOperation() const { return m_selectedOperation; }

    //! Returns whether mesh order has been swappped or not
    bool isSwapped() const { return m_isSwapped; }

protected slots:

    void unionSelected();
    void intersectSelected();
    void diffSelected();
    void symDiffSelected();
    void swap();

protected:
    CSG_OPERATION m_selectedOperation;
    bool m_isSwapped;
};

#endif  // ECV_CORK_DLG_HEADER
