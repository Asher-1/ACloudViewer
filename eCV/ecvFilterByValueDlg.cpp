// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvFilterByValueDlg.h"

ccFilterByValueDlg::ccFilterByValueDlg(double minRange,
                                       double maxRange,
                                       double minVal /*=-1.0e9*/,
                                       double maxVal /*=1.0e9*/,
                                       QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::FilterByValueDialog(), m_mode(CANCEL) {
    setupUi(this);

    minDoubleSpinBox->setRange(minVal, maxVal);
    maxDoubleSpinBox->setRange(minVal, maxVal);
    minDoubleSpinBox->setValue(minRange);
    maxDoubleSpinBox->setValue(maxRange);

    connect(exportPushButton, &QAbstractButton::clicked, this,
            &ccFilterByValueDlg::onExport);
    connect(splitPushButton, &QAbstractButton::clicked, this,
            &ccFilterByValueDlg::onSplit);
}
