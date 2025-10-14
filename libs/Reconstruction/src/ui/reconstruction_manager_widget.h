// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UI_RECONSTRUCTION_MANAGER_WIDGET_H_
#define COLMAP_SRC_UI_RECONSTRUCTION_MANAGER_WIDGET_H_

#include <QtWidgets>

#include "base/reconstruction_manager.h"

namespace colmap {

class ReconstructionManagerWidget : public QComboBox {
public:
    const static size_t kNewestReconstructionIdx;

    ReconstructionManagerWidget(
            QWidget* parent,
            const ReconstructionManager* reconstruction_manager);

    void Update();

    size_t SelectedReconstructionIdx() const;
    void SelectReconstruction(const size_t idx);

private:
    const ReconstructionManager* reconstruction_manager_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_UI_RECONSTRUCTION_MANAGER_WIDGET_H_
