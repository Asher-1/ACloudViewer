// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QDialog>

// GUI
#include "ui_openRDBDlg.h"

enum CC_RDB_OPEN_DLG_TYPES {
    RDB_OPEN_DLG_None = 0,
    RDB_OPEN_DLG_XYZ = 1,
    RDB_OPEN_DLG_NORM = 2,
    RDB_OPEN_DLG_RGB = 3,
    RDB_OPEN_DLG_Grey = 4,
    RDB_OPEN_DLG_Scalar = 5,
};
const unsigned RDB_OPEN_DLG_TYPES_NUMBER = 6;
const char RDB_OPEN_DLG_TYPES_NAMES[RDB_OPEN_DLG_TYPES_NUMBER][24] = {
        "Ignore", "XYZ", "Normals", "RGB", "Grey", "Scalar",
};
//! RDB Open dialog
class RDBOpenDialog : public QDialog, public Ui::RDBOpenDlg {
    Q_OBJECT

public:
    //! Default constructor
    explicit RDBOpenDialog(QWidget* parent = nullptr);
};
