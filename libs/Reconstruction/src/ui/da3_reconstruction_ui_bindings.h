// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QObject>

#include "controllers/da3_depth_controller.h"

namespace colmap {

// Widget handles for DA3-related automatic reconstruction UI controls.
struct DA3ReconstructionUiControls {
    QComboBox* sparse_mode_cb = nullptr;
    QComboBox* stereo_mode_cb = nullptr;
    QComboBox* da3_sparse_model_cb = nullptr;
    QComboBox* da3_sparse_quant_cb = nullptr;
    QComboBox* da3_stereo_model_cb = nullptr;
    QComboBox* da3_stereo_quant_cb = nullptr;
    QLabel* da3_sparse_model_label = nullptr;
    QLabel* da3_sparse_quant_label = nullptr;
    QLabel* da3_stereo_model_label = nullptr;
    QLabel* da3_stereo_quant_label = nullptr;
    QCheckBox* da3_force_recompute_cb = nullptr;
    QLabel* da3_force_recompute_label = nullptr;
    QCheckBox* da3_skip_geometric_refine_cb = nullptr;
    QLabel* da3_skip_geometric_refine_label = nullptr;
    QCheckBox* dense_cb = nullptr;
    QCheckBox* fused_point_filter_cb = nullptr;
    QDoubleSpinBox* fused_voxel_size_spin = nullptr;
    QLabel* fused_voxel_size_label = nullptr;
    QLabel* da3_hybrid_hint_label = nullptr;
};

// Shared combobox linkage: sparse/stereo modes, per-step model+quant
// visibility.
class DA3ReconstructionUiBindings {
public:
    static constexpr int kSparseColmap = 0;
    static constexpr int kSparseDa3 = 1;
    static constexpr int kStereoPatchMatch = 0;
    static constexpr int kStereoDa3 = 1;

    static constexpr int kSparseModelBase = 0;
    static constexpr int kSparseModelLarge = 1;
    static constexpr int kSparseModelGiant = 2;
    static constexpr int kSparseModelNestedMetric = 3;
    static constexpr int kSparseModelNestedAnyView = 4;

    static constexpr int kStereoModelNestedMetric = 0;
    static constexpr int kStereoModelNestedAnyView = 1;

    static void InitStereoModeComboBox(QComboBox* stereo_mode_cb);
    static void InitSparseModeComboBox(QComboBox* sparse_mode_cb);
    static void InitSparseModelComboBox(QComboBox* da3_sparse_model_cb);
    static void InitStereoModelComboBox(QComboBox* da3_stereo_model_cb);
    static void Install(const DA3ReconstructionUiControls& controls,
                        QObject* context);
    static void Sync(const DA3ReconstructionUiControls& controls);
    static void ApplyDa3FusedPointFilterDefaults(
            const DA3ReconstructionUiControls& controls);
    static void ApplyHybridDenseTooltips(
            const DA3ReconstructionUiControls& controls,
            const std::string& image_path = {});
    static void UpdateHybridDenseHint(
            const DA3ReconstructionUiControls& controls,
            const std::string& image_path = {});
    // Hide/disable DA3 pipeline options when libAICore is unavailable.
    static void SetAICoreAvailable(const DA3ReconstructionUiControls& controls,
                                   bool available);

    static bool IsNestedSparseModelIndex(int model_index);
    static DA3ModelType SparseModelTypeFromIndex(int model_index);
    static DA3ModelType StereoModelTypeFromIndex(int model_index);
    static DA3QuantType QuantTypeFromComboText(const QString& text);

private:
    static void PopulateQuantComboBox(QComboBox* da3_quant_cb,
                                      int model_index,
                                      bool stereo_step);
    static void SetSparseDa3ItemEnabled(QComboBox* sparse_mode_cb,
                                        bool enabled);
    static void SetStereoDa3ItemEnabled(QComboBox* stereo_mode_cb,
                                        bool enabled);
    static void UpdateSparseModelVisibility(
            const DA3ReconstructionUiControls& controls, bool sparse_da3);
    static void UpdateStereoModelVisibility(
            const DA3ReconstructionUiControls& controls, bool stereo_da3);
    static void OnSparseModeChanged(const DA3ReconstructionUiControls& controls,
                                    int sparse_index);
    static void OnStereoModeChanged(const DA3ReconstructionUiControls& controls,
                                    int stereo_index);
    static void OnSparseModelChanged(
            const DA3ReconstructionUiControls& controls, int model_index);
    static void OnStereoModelChanged(
            const DA3ReconstructionUiControls& controls, int model_index);
    static void OnDenseChanged(const DA3ReconstructionUiControls& controls,
                               bool dense_enabled);
};

}  // namespace colmap
