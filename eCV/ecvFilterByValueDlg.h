// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_FILTER_BY_VALUE_DIALOG_HEADER
#define ECV_FILTER_BY_VALUE_DIALOG_HEADER

#include <ui_filterByValueDlg.h>

// Qt
#include <QDialog>

//! Dialog to sepcify a range of SF values and how the corresponding points
//! should be extracted
class ccFilterByValueDlg : public QDialog, public Ui::FilterByValueDialog {
    Q_OBJECT

public:
    //! Default constructor
    ccFilterByValueDlg(double minRange,
                       double maxRange,
                       double minVal = -1.0e9,
                       double maxVal = 1.0e9,
                       QWidget* parent = 0);

    //! Mode
    enum Mode { EXPORT, SPLIT, CANCEL };

    //! Returns the selected mode
    Mode mode() const { return m_mode; }

protected slots:

    void onExport() {
        m_mode = EXPORT;
        accept();
    }
    void onSplit() {
        m_mode = SPLIT;
        accept();
    }

protected:
    Mode m_mode;
};

#endif  // ECV_FILTER_BY_VALUE_DIALOG_HEADER
