// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UI_FEATURE_MATCHING_WIDGET_H_
#define COLMAP_SRC_UI_FEATURE_MATCHING_WIDGET_H_

#include <QtCore>
#include <QtWidgets>

#include "util/misc.h"
#include "util/option_manager.h"

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

#endif  // COLMAP_SRC_UI_FEATURE_MATCHING_WIDGET_H_
