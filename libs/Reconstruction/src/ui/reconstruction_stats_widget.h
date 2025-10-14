// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UI_RECONSTRUCTION_STATS_WIDGET_H_
#define COLMAP_SRC_UI_RECONSTRUCTION_STATS_WIDGET_H_

#include <QtWidgets>

#include "base/reconstruction.h"

namespace colmap {

class ReconstructionStatsWidget : public QWidget {
public:
    explicit ReconstructionStatsWidget(QWidget* parent);

    void Show(const Reconstruction& reconstruction);

private:
    void AddStatistic(const QString& header, const QString& content);

    QTableWidget* stats_table_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_RECONSTRUCTION_STATS_WIDGET_H_
