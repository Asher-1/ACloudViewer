// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtWidgets>

namespace colmap {
class ReconstructionManager;
}

namespace cloudViewer {
class ReconstructionManagerWidget : public QComboBox {
public:
    const static size_t kNewestReconstructionIdx;

    ReconstructionManagerWidget(
            QWidget* parent,
            const colmap::ReconstructionManager* reconstruction_manager);

    void Update();

    size_t SelectedReconstructionIdx() const;
    void SelectReconstruction(const size_t idx);

private:
    const colmap::ReconstructionManager* reconstruction_manager_;
};

}  // namespace cloudViewer
