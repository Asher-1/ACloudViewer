// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QFACET_CLASSIFICATION_PARAMS_DLG_HEADER
#define QFACET_CLASSIFICATION_PARAMS_DLG_HEADER

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

#endif  // QFACET_CLASSIFICATION_PARAMS_DLG_HEADER
