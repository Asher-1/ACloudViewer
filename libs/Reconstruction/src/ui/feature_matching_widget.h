// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "controllers/option_manager.h"
#include "util/misc.h"

namespace colmap {

class FeatureMatchingWidget : public QWidget {
public:
    FeatureMatchingWidget(QWidget* parent, OptionManager* options);

private:
    void showEvent(QShowEvent* event);
    void hideEvent(QHideEvent* event);

    QWidget* parent_;
    QTabWidget* tab_widget_;
};

}  // namespace colmap
