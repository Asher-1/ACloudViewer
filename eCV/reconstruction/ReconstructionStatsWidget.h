// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtWidgets>

namespace colmap {
class Reconstruction;
}

namespace cloudViewer {

class ReconstructionStatsWidget : public QWidget {
public:
    explicit ReconstructionStatsWidget(QWidget* parent);

    void Show(const colmap::Reconstruction& reconstruction);

private:
    void AddStatistic(const QString& header, const QString& content);

    QTableWidget* stats_table_;
};

}  // namespace cloudViewer
