// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_PICK_ONE_ELEMENT_DLG_HEADER
#define ECV_PICK_ONE_ELEMENT_DLG_HEADER

#include "CVAppCommon.h"

// Qt
#include <QDialog>

class Ui_PickOneElementDialog;

//! Minimal dialog to pick one element in a list (combox box)
class CVAPPCOMMON_LIB_API ccPickOneElementDlg : public QDialog {
    Q_OBJECT

public:
    //! Default constructor
    ccPickOneElementDlg(const QString &label,
                        const QString &windowTitle = QString(),
                        QWidget *parent = nullptr);

    //! Destructor
    ~ccPickOneElementDlg() override;

    //! Add an element to the combo box
    void addElement(const QString &elementName);
    //! Sets the combo box default index
    void setDefaultIndex(int index);
    //! Returns the combo box current index (after completion)
    int getSelectedIndex();

protected:
    //! Associated UI
    Ui_PickOneElementDialog *m_ui;
};

#endif  // ECV_PICK_ONE_ELEMENT_DLG_HEADER
