// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QObject>

#include "controllers/da3_depth_controller.h"

namespace colmap {

// Widget handles for DA3-related automatic reconstruction UI controls.
struct DA3ReconstructionUiControls {
    QComboBox* sparse_mode_cb = nullptr;
    QComboBox* stereo_mode_cb = nullptr;
    QComboBox* da3_model_cb = nullptr;
    QComboBox* da3_quant_cb = nullptr;
    QCheckBox* dense_cb = nullptr;
};

// Shared combobox linkage: nested-only DA3 stereo, quant refresh, dense gating.
class DA3ReconstructionUiBindings {
public:
    static constexpr int kSparseColmap = 0;
    static constexpr int kSparseDa3 = 1;
    static constexpr int kStereoPatchMatch = 0;
    static constexpr int kStereoDa3 = 1;
    static constexpr int kModelNestedMetric = 3;
    static constexpr int kModelNestedAnyView = 4;

    static void InitStereoComboBox(QComboBox* stereo_mode_cb);
    static void Install(const DA3ReconstructionUiControls& controls,
                        QObject* context);
    static void Sync(const DA3ReconstructionUiControls& controls);

    static bool IsNestedModelIndex(int model_index);
    static DA3ModelType ModelTypeFromIndex(int model_index);

private:
    static void PopulateQuantComboBox(QComboBox* da3_quant_cb, int model_index);
    static void SetStereoDa3ItemEnabled(QComboBox* stereo_mode_cb, bool enabled);
    static void OnModelChanged(const DA3ReconstructionUiControls& controls,
                               int model_index);
    static void OnStereoModeChanged(const DA3ReconstructionUiControls& controls,
                                    int stereo_index);
    static void OnDenseChanged(const DA3ReconstructionUiControls& controls,
                               bool dense_enabled);
};

}  // namespace colmap
