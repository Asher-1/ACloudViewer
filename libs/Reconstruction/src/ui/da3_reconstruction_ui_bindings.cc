// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ui/da3_reconstruction_ui_bindings.h"

#include <QStandardItemModel>

#include "controllers/da3_depth_controller.h"
#include "controllers/da3_pipeline_defaults.h"
#include "util/misc.h"

namespace colmap {
namespace {

bool g_aicore_available = true;

void BlockCombo(QComboBox* cb, int index) {
    if (!cb) {
        return;
    }
    cb->blockSignals(true);
    cb->setCurrentIndex(index);
    cb->blockSignals(false);
}

void SetRowVisible(QLabel* label, QComboBox* combo, bool visible) {
    if (label) {
        label->setVisible(visible);
    }
    if (combo) {
        combo->setVisible(visible);
    }
}

void SetRowVisible(QLabel* label, QCheckBox* checkbox, bool visible) {
    if (label) {
        label->setVisible(visible);
    }
    if (checkbox) {
        checkbox->setVisible(visible);
    }
}

void EnableFusedPointFilterForDa3(const DA3ReconstructionUiControls& controls) {
    if (!controls.fused_point_filter_cb) {
        return;
    }
    controls.fused_point_filter_cb->blockSignals(true);
    controls.fused_point_filter_cb->setChecked(true);
    controls.fused_point_filter_cb->blockSignals(false);
    if (controls.fused_voxel_size_spin) {
        controls.fused_voxel_size_spin->setEnabled(true);
    }
    if (controls.fused_voxel_size_label) {
        controls.fused_voxel_size_label->setEnabled(true);
    }
}

}  // namespace

void DA3ReconstructionUiBindings::InitStereoModeComboBox(
    QComboBox* stereo_mode_cb) {
    if (!stereo_mode_cb) {
        return;
    }
    auto* stereo_model = new QStandardItemModel(stereo_mode_cb);
    stereo_model->appendRow(new QStandardItem("COLMAP PatchMatch"));
    stereo_model->appendRow(
        new QStandardItem("DA3 Depth Inference (Nested only)"));
    stereo_mode_cb->setModel(stereo_model);
    stereo_mode_cb->setCurrentIndex(PreferDA3OverColmapPatchMatch()
                                            ? kStereoDa3
                                            : kStereoPatchMatch);
}

void DA3ReconstructionUiBindings::InitSparseModeComboBox(
    QComboBox* sparse_mode_cb) {
    if (!sparse_mode_cb) {
        return;
    }
    sparse_mode_cb->clear();
    sparse_mode_cb->addItem("COLMAP (native SfM)");
    sparse_mode_cb->addItem("DA3 hybrid (≥3 img: COLMAP poses)");
    sparse_mode_cb->setCurrentIndex(PreferDA3OverColmapPatchMatch()
                                            ? kSparseDa3
                                            : kSparseColmap);
}

void DA3ReconstructionUiBindings::InitSparseModelComboBox(
    QComboBox* da3_sparse_model_cb) {
    if (!da3_sparse_model_cb) {
        return;
    }
    da3_sparse_model_cb->clear();
    da3_sparse_model_cb->addItem("Base (ViT-B, fastest)");
    da3_sparse_model_cb->addItem("Large (ViT-L)");
    da3_sparse_model_cb->addItem("Giant (ViT-G, best quality)");
    da3_sparse_model_cb->addItem("Nested Metric");
    da3_sparse_model_cb->addItem("Nested AnyView");
    da3_sparse_model_cb->setCurrentIndex(
            PreferDA3OverColmapPatchMatch() ? kSparseModelNestedAnyView
                                            : kSparseModelBase);
}

void DA3ReconstructionUiBindings::InitStereoModelComboBox(
    QComboBox* da3_stereo_model_cb) {
    if (!da3_stereo_model_cb) {
        return;
    }
    da3_stereo_model_cb->clear();
    da3_stereo_model_cb->addItem("Nested Metric");
    da3_stereo_model_cb->addItem("Nested AnyView");
    da3_stereo_model_cb->setCurrentIndex(kStereoModelNestedAnyView);
}

bool DA3ReconstructionUiBindings::IsNestedSparseModelIndex(int model_index) {
    return model_index == kSparseModelNestedMetric ||
           model_index == kSparseModelNestedAnyView;
}

DA3ModelType DA3ReconstructionUiBindings::SparseModelTypeFromIndex(
    int model_index) {
    switch (model_index) {
        case kSparseModelBase:
            return DA3ModelType::BASE;
        case kSparseModelLarge:
            return DA3ModelType::LARGE;
        case kSparseModelGiant:
            return DA3ModelType::GIANT;
        case kSparseModelNestedMetric:
            return DA3ModelType::NESTED_METRIC;
        case kSparseModelNestedAnyView:
            return DA3ModelType::NESTED_ANYVIEW;
        default:
            return DA3ModelType::BASE;
    }
}

DA3ModelType DA3ReconstructionUiBindings::StereoModelTypeFromIndex(
    int model_index) {
    switch (model_index) {
        case kStereoModelNestedMetric:
            return DA3ModelType::NESTED_METRIC;
        case kStereoModelNestedAnyView:
        default:
            return DA3ModelType::NESTED_ANYVIEW;
    }
}

DA3QuantType DA3ReconstructionUiBindings::QuantTypeFromComboText(
    const QString& text) {
    if (text.startsWith("Q4_K")) {
        return DA3QuantType::Q4_K;
    }
    if (text.startsWith("F16")) {
        return DA3QuantType::F16;
    }
    if (text.startsWith("F32")) {
        return DA3QuantType::F32;
    }
    return DA3QuantType::Q8_0;
}

void DA3ReconstructionUiBindings::PopulateQuantComboBox(
    QComboBox* da3_quant_cb, int model_index, bool stereo_step) {
    if (!da3_quant_cb) {
        return;
    }
    const DA3ModelType model_type =
        stereo_step ? StereoModelTypeFromIndex(model_index)
                    : SparseModelTypeFromIndex(model_index);
    da3_quant_cb->blockSignals(true);
    da3_quant_cb->clear();
    const auto supported = DA3SupportedQuantTypes(model_type);
    for (const auto qt : supported) {
        switch (qt) {
            case DA3QuantType::Q8_0:
                da3_quant_cb->addItem("Q8_0 (recommended)");
                break;
            case DA3QuantType::Q4_K:
                da3_quant_cb->addItem("Q4_K (smallest)");
                break;
            case DA3QuantType::F16:
                da3_quant_cb->addItem("F16 (half precision)");
                break;
            case DA3QuantType::F32:
                da3_quant_cb->addItem("F32 (full precision)");
                break;
        }
    }
    da3_quant_cb->blockSignals(false);
}

void DA3ReconstructionUiBindings::SetSparseDa3ItemEnabled(
    QComboBox* sparse_mode_cb, bool enabled) {
    if (!sparse_mode_cb) {
        return;
    }
    if (auto* sparse_model =
            qobject_cast<QStandardItemModel*>(sparse_mode_cb->model())) {
        if (QStandardItem* da3_item = sparse_model->item(kSparseDa3)) {
            da3_item->setEnabled(enabled);
        }
    }
}

void DA3ReconstructionUiBindings::SetStereoDa3ItemEnabled(
    QComboBox* stereo_mode_cb, bool enabled) {
    if (!stereo_mode_cb) {
        return;
    }
    if (auto* stereo_model =
            qobject_cast<QStandardItemModel*>(stereo_mode_cb->model())) {
        if (QStandardItem* da3_item = stereo_model->item(kStereoDa3)) {
            da3_item->setEnabled(enabled);
        }
    }
}

void DA3ReconstructionUiBindings::SetAICoreAvailable(
    const DA3ReconstructionUiControls& controls, bool available) {
    g_aicore_available = available;

    SetSparseDa3ItemEnabled(controls.sparse_mode_cb, available);
    const bool dense_enabled =
        !controls.dense_cb || controls.dense_cb->isChecked();
    SetStereoDa3ItemEnabled(controls.stereo_mode_cb,
                            available && dense_enabled);

    if (!available) {
        if (controls.sparse_mode_cb &&
            controls.sparse_mode_cb->currentIndex() == kSparseDa3) {
            BlockCombo(controls.sparse_mode_cb, kSparseColmap);
        }
        if (controls.stereo_mode_cb &&
            controls.stereo_mode_cb->currentIndex() == kStereoDa3) {
            BlockCombo(controls.stereo_mode_cb, kStereoPatchMatch);
        }
        if (controls.da3_hybrid_hint_label) {
            controls.da3_hybrid_hint_label->hide();
        }
    }

    Sync(controls);
}

void DA3ReconstructionUiBindings::ApplyPreferDa3Defaults(
        const DA3ReconstructionUiControls& controls) {
    if (!PreferDA3OverColmapPatchMatch()) {
        return;
    }
    if (controls.sparse_mode_cb) {
        BlockCombo(controls.sparse_mode_cb, kSparseDa3);
    }
    if (controls.stereo_mode_cb) {
        BlockCombo(controls.stereo_mode_cb, kStereoDa3);
    }
    if (controls.dense_cb && !controls.dense_cb->isChecked()) {
        controls.dense_cb->blockSignals(true);
        controls.dense_cb->setChecked(true);
        controls.dense_cb->blockSignals(false);
    }
    if (controls.da3_sparse_model_cb) {
        BlockCombo(controls.da3_sparse_model_cb, kSparseModelNestedAnyView);
    }
    if (controls.da3_stereo_model_cb) {
        BlockCombo(controls.da3_stereo_model_cb, kStereoModelNestedAnyView);
    }
    if (controls.use_gpu_cb && controls.use_gpu_cb->isChecked()) {
        controls.use_gpu_cb->blockSignals(true);
        controls.use_gpu_cb->setChecked(false);
        controls.use_gpu_cb->blockSignals(false);
    }
    if (controls.use_gpu) {
        *controls.use_gpu = false;
    }
    Sync(controls);
}

void DA3ReconstructionUiBindings::UpdateSparseModelVisibility(
    const DA3ReconstructionUiControls& controls, bool sparse_da3) {
    SetRowVisible(controls.da3_sparse_model_label, controls.da3_sparse_model_cb,
                  sparse_da3);
    SetRowVisible(controls.da3_sparse_quant_label, controls.da3_sparse_quant_cb,
                  sparse_da3);
}

static void UpdateForceRecomputeVisibility(
    const DA3ReconstructionUiControls& controls, bool show) {
    SetRowVisible(controls.da3_force_recompute_label,
                  controls.da3_force_recompute_cb, show);
}

static void UpdateSkipGeometricRefineVisibility(
    const DA3ReconstructionUiControls& controls, bool show) {
    SetRowVisible(controls.da3_skip_geometric_refine_label,
                  controls.da3_skip_geometric_refine_cb, show);
}

static void UpdateHybridDenseOptionVisibility(
    const DA3ReconstructionUiControls& controls, bool show) {
    UpdateForceRecomputeVisibility(controls, show);
    UpdateSkipGeometricRefineVisibility(controls, show);
}

void DA3ReconstructionUiBindings::UpdateStereoModelVisibility(
    const DA3ReconstructionUiControls& controls, bool stereo_da3) {
    SetRowVisible(controls.da3_stereo_model_label, controls.da3_stereo_model_cb,
                  stereo_da3);
    SetRowVisible(controls.da3_stereo_quant_label, controls.da3_stereo_quant_cb,
                  stereo_da3);
}

void DA3ReconstructionUiBindings::OnSparseModeChanged(
    const DA3ReconstructionUiControls& controls, int sparse_index) {
    UpdateSparseModelVisibility(controls, sparse_index == kSparseDa3);
    const bool stereo_da3 = controls.stereo_mode_cb &&
                            controls.stereo_mode_cb->currentIndex() == kStereoDa3;
    UpdateForceRecomputeVisibility(controls,
                                   sparse_index == kSparseDa3 || stereo_da3);
    UpdateSkipGeometricRefineVisibility(
        controls, sparse_index == kSparseDa3 && stereo_da3 &&
                      (!controls.dense_cb || controls.dense_cb->isChecked()));
    UpdateHybridDenseHint(controls);
}

void DA3ReconstructionUiBindings::OnStereoModeChanged(
    const DA3ReconstructionUiControls& controls, int stereo_index) {
    if (stereo_index == kStereoDa3 && controls.dense_cb &&
        !controls.dense_cb->isChecked()) {
        controls.dense_cb->blockSignals(true);
        controls.dense_cb->setChecked(true);
        controls.dense_cb->blockSignals(false);
    }
    if (stereo_index == kStereoDa3) {
        EnableFusedPointFilterForDa3(controls);
    }
    UpdateStereoModelVisibility(controls, stereo_index == kStereoDa3);
    const bool sparse_da3 = controls.sparse_mode_cb &&
                            controls.sparse_mode_cb->currentIndex() == kSparseDa3;
    UpdateForceRecomputeVisibility(controls,
                                   sparse_da3 || stereo_index == kStereoDa3);
    UpdateSkipGeometricRefineVisibility(
        controls, sparse_da3 && stereo_index == kStereoDa3 &&
                      (!controls.dense_cb || controls.dense_cb->isChecked()));
    UpdateHybridDenseHint(controls);
}

void DA3ReconstructionUiBindings::OnSparseModelChanged(
    const DA3ReconstructionUiControls& controls, int model_index) {
    PopulateQuantComboBox(controls.da3_sparse_quant_cb, model_index, false);
}

void DA3ReconstructionUiBindings::OnStereoModelChanged(
    const DA3ReconstructionUiControls& controls, int model_index) {
    PopulateQuantComboBox(controls.da3_stereo_quant_cb, model_index, true);
}

void DA3ReconstructionUiBindings::OnDenseChanged(
    const DA3ReconstructionUiControls& controls, bool dense_enabled) {
    if (controls.stereo_mode_cb) {
        controls.stereo_mode_cb->setEnabled(dense_enabled);
    }

    if (!dense_enabled) {
        if (controls.stereo_mode_cb &&
            controls.stereo_mode_cb->currentIndex() == kStereoDa3) {
            BlockCombo(controls.stereo_mode_cb, kStereoPatchMatch);
        }
        SetStereoDa3ItemEnabled(controls.stereo_mode_cb, false);
        UpdateStereoModelVisibility(controls, false);
        const bool sparse_da3 = controls.sparse_mode_cb &&
                                controls.sparse_mode_cb->currentIndex() ==
                                    kSparseDa3;
        UpdateForceRecomputeVisibility(controls, sparse_da3);
        UpdateSkipGeometricRefineVisibility(controls, false);
        return;
    }

    SetStereoDa3ItemEnabled(controls.stereo_mode_cb, g_aicore_available);
    const bool stereo_da3 = controls.stereo_mode_cb &&
                            controls.stereo_mode_cb->currentIndex() == kStereoDa3;
    UpdateStereoModelVisibility(controls, stereo_da3);
    const bool sparse_da3 = controls.sparse_mode_cb &&
                            controls.sparse_mode_cb->currentIndex() ==
                                kSparseDa3;
    UpdateForceRecomputeVisibility(controls, sparse_da3 || stereo_da3);
    UpdateSkipGeometricRefineVisibility(
        controls, sparse_da3 && stereo_da3 &&
                      (!controls.dense_cb || controls.dense_cb->isChecked()));
}

void DA3ReconstructionUiBindings::ApplyDa3FusedPointFilterDefaults(
    const DA3ReconstructionUiControls& controls) {
    if (controls.stereo_mode_cb &&
        controls.stereo_mode_cb->currentIndex() == kStereoDa3) {
        EnableFusedPointFilterForDa3(controls);
    }
}

void DA3ReconstructionUiBindings::Sync(
    const DA3ReconstructionUiControls& controls) {
    const int sparse_index =
        controls.sparse_mode_cb ? controls.sparse_mode_cb->currentIndex()
                                : kSparseColmap;
    const int sparse_model_index =
        controls.da3_sparse_model_cb ? controls.da3_sparse_model_cb->currentIndex()
                                     : kSparseModelBase;
    const int stereo_model_index =
        controls.da3_stereo_model_cb ? controls.da3_stereo_model_cb->currentIndex()
                                     : kStereoModelNestedAnyView;

    OnSparseModelChanged(controls, sparse_model_index);
    OnStereoModelChanged(controls, stereo_model_index);
    OnSparseModeChanged(controls, sparse_index);
    OnDenseChanged(controls,
                   !controls.dense_cb || controls.dense_cb->isChecked());
    if (controls.stereo_mode_cb) {
        OnStereoModeChanged(controls, controls.stereo_mode_cb->currentIndex());
    }
}

void DA3ReconstructionUiBindings::Install(
    const DA3ReconstructionUiControls& controls, QObject* context) {
    if (!context) {
        return;
    }

    if (controls.sparse_mode_cb) {
        QObject::connect(
            controls.sparse_mode_cb,
            QOverload<int>::of(&QComboBox::currentIndexChanged), context,
            [controls](int index) { OnSparseModeChanged(controls, index); });
    }

    if (controls.stereo_mode_cb) {
        QObject::connect(
            controls.stereo_mode_cb,
            QOverload<int>::of(&QComboBox::currentIndexChanged), context,
            [controls](int index) { OnStereoModeChanged(controls, index); });
    }

    if (controls.da3_sparse_model_cb) {
        QObject::connect(
            controls.da3_sparse_model_cb,
            QOverload<int>::of(&QComboBox::currentIndexChanged), context,
            [controls](int index) { OnSparseModelChanged(controls, index); });
    }

    if (controls.da3_stereo_model_cb) {
        QObject::connect(
            controls.da3_stereo_model_cb,
            QOverload<int>::of(&QComboBox::currentIndexChanged), context,
            [controls](int index) { OnStereoModelChanged(controls, index); });
    }

    if (controls.dense_cb) {
        QObject::connect(controls.dense_cb, &QCheckBox::toggled, context,
                         [controls](bool checked) {
                             OnDenseChanged(controls, checked);
                         });
    }

    ApplyHybridDenseTooltips(controls);
    UpdateHybridDenseHint(controls);
    Sync(controls);
}

void DA3ReconstructionUiBindings::UpdateHybridDenseHint(
    const DA3ReconstructionUiControls& controls,
    const std::string& image_path) {
    if (!controls.da3_hybrid_hint_label) {
        return;
    }

    const bool sparse_da3 = controls.sparse_mode_cb &&
                            controls.sparse_mode_cb->currentIndex() == kSparseDa3;
    const bool stereo_da3 = controls.stereo_mode_cb &&
                            controls.stereo_mode_cb->currentIndex() == kStereoDa3;
    const bool dense_enabled =
        !controls.dense_cb || controls.dense_cb->isChecked();

    if (!sparse_da3 || !stereo_da3 || !dense_enabled) {
        controls.da3_hybrid_hint_label->hide();
        return;
    }

    size_t num_images = 0;
    if (!image_path.empty() && ExistsDir(image_path)) {
        num_images = CountDA3Images(image_path);
    }

    QString hint;
    if (num_images >= static_cast<size_t>(kDA3ColmapSparseAutoMinViews)) {
        hint = QObject::tr(
            "Hybrid dense (%1 images): sparse camera poses from COLMAP SfM; "
            "DA3 supplies metric depth priors only (not per-view poses). "
            "Enable “Skip geometric refine” for fastest StereoFusion "
            "(auto-fallback if quality is low).")
                   .arg(static_cast<qulonglong>(num_images));
    } else if (num_images > 0) {
        hint = QObject::tr(
            "≤2 images: DA3 depth+pose unified mode (no COLMAP hybrid). "
            "%1 image(s) detected.")
                   .arg(static_cast<qulonglong>(num_images));
    } else {
        hint = QObject::tr(
            "≥3 images with DA3 stereo: hybrid mode uses COLMAP sparse poses "
            "+ DA3 metric depth (poses are not estimated by DA3).");
    }

    controls.da3_hybrid_hint_label->setText(hint);
    controls.da3_hybrid_hint_label->show();
}

void DA3ReconstructionUiBindings::ApplyHybridDenseTooltips(
    const DA3ReconstructionUiControls& controls,
    const std::string& image_path) {
    if (controls.sparse_mode_cb) {
        QString sparse_tip = QObject::tr(
            "DA3 hybrid (recommended with DA3 stereo): enables the hybrid "
            "dense pipeline. With 3+ images, camera poses always come from "
            "COLMAP SfM; DA3 only provides metric depth priors (per-view DA3 "
            "poses are not globally consistent). With ≤2 images, DA3 depth+pose "
            "is used end-to-end.");
        if (!image_path.empty() && ExistsDir(image_path)) {
            const size_t num_images = CountDA3Images(image_path);
            if (num_images >= static_cast<size_t>(kDA3ColmapSparseAutoMinViews)) {
                sparse_tip += QObject::tr(
                    "\n\nCurrent folder: %1 images → COLMAP poses + DA3 depth.")
                                  .arg(static_cast<qulonglong>(num_images));
            }
        }
        controls.sparse_mode_cb->setToolTip(sparse_tip);
    }
    if (controls.stereo_mode_cb) {
        controls.stereo_mode_cb->setToolTip(
            QObject::tr(
                "Nested AnyView/Metric depth replaces COLMAP photometric "
                "PatchMatch. With hybrid sparse+stereo (3+ images), use "
                "“Skip geometric refine” for DA3 voxel fusion on priors "
                "(fastest; auto-fallback). "
                "DA3_FULL_PATCHMATCH=1 forces full NCC re-optimization."));
    }
    if (controls.da3_skip_geometric_refine_cb) {
        controls.da3_skip_geometric_refine_cb->setToolTip(
            QObject::tr(
                "Skip PatchMatch geometric refinement and fuse DA3 metric "
                "depth priors directly. Falls back to geometric refine "
                "automatically if the fused point count is too low. "
                "Requires Sparse=DA3 hybrid and Stereo=DA3 with Dense "
                "enabled (3+ images)."));
    }
    UpdateHybridDenseHint(controls, image_path);
}

}  // namespace colmap
