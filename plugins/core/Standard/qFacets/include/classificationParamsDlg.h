// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

#include "ui_classificationParamsDlg.h"

//! Dialog for orientation-based classification of facets (qFacets plugin)
class ClassificationParamsDlg : public QDialog,
                                public Ui::ClassificationParamsDlg {
public:
    //! Default constructor
    ClassificationParamsDlg(QWidget* parent = 0)
        : QDialog(parent, Qt::Tool), Ui::ClassificationParamsDlg() {
        setupUi(this);
    }
};
