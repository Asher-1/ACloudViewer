// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccTreeIsoDlg.h"

#include <ecvOctree.h>
#include <qTreeIso.h>

ccTreeIsoDlg::ccTreeIsoDlg(QWidget* parent)
    : QDialog(parent), Ui::TreeIsoDialog() {
    setupUi(this);
    setWindowFlags(Qt::Tool /*Qt::Dialog | Qt::WindowStaysOnTopHint*/);
}
