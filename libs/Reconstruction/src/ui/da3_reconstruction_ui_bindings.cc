// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ui/da3_reconstruction_ui_bindings.h"

#include <QStandardItemModel>

#include "controllers/da3_depth_controller.h"

namespace colmap {
namespace {

void BlockCombo(QComboBox* cb, int index) {
    if (!cb) {
        return;
    }
    cb->blockSignals(true);
    cb->setCurrentIndex(index);
    cb->blockSignals(false);
}

}  // namespace

void DA3ReconstructionUiBindings::InitStereoComboBox(QComboBox* stereo_mode_cb) {
    if (!stereo_mode_cb) {
        return;
    }
    auto* stereo_model = new QStandardItemModel(stereo_mode_cb);
    stereo_model->appendRow(new QStandardItem("COLMAP PatchMatch"));
    stereo_model->appendRow(
        new QStandardItem("DA3 Depth Inference (Nested only)"));
    stereo_mode_cb->setModel(stereo_model);
    stereo_mode_cb->setCurrentIndex(kStereoPatchMatch);
}

bool DA3ReconstructionUiBindings::IsNestedModelIndex(int model_index) {
    return model_index == kModelNestedMetric ||
           model_index == kModelNestedAnyView;
}

DA3ModelType DA3ReconstructionUiBindings::ModelTypeFromIndex(int model_index) {
    switch (model_index) {
        case 0:
            return DA3ModelType::BASE;
        case 1:
            return DA3ModelType::LARGE;
        case 2:
            return DA3ModelType::GIANT;
        case kModelNestedMetric:
            return DA3ModelType::NESTED_METRIC;
        case kModelNestedAnyView:
            return DA3ModelType::NESTED_ANYVIEW;
        default:
            return DA3ModelType::BASE;
    }
}

void DA3ReconstructionUiBindings::PopulateQuantComboBox(QComboBox* da3_quant_cb,
                                                        int model_index) {
    if (!da3_quant_cb) {
        return;
    }
    da3_quant_cb->blockSignals(true);
    da3_quant_cb->clear();
    const auto supported =
        DA3SupportedQuantTypes(ModelTypeFromIndex(model_index));
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

void DA3ReconstructionUiBindings::OnModelChanged(
    const DA3ReconstructionUiControls& controls, int model_index) {
    PopulateQuantComboBox(controls.da3_quant_cb, model_index);

    const bool nested = IsNestedModelIndex(model_index);
    const bool dense_enabled =
        !controls.dense_cb || controls.dense_cb->isChecked();

    SetStereoDa3ItemEnabled(controls.stereo_mode_cb, nested && dense_enabled);

    if (!nested && controls.stereo_mode_cb &&
        controls.stereo_mode_cb->currentIndex() == kStereoDa3) {
        BlockCombo(controls.stereo_mode_cb, kStereoPatchMatch);
    }
}

void DA3ReconstructionUiBindings::OnStereoModeChanged(
    const DA3ReconstructionUiControls& controls, int stereo_index) {
    if (stereo_index != kStereoDa3) {
        return;
    }

    if (controls.dense_cb && !controls.dense_cb->isChecked()) {
        controls.dense_cb->blockSignals(true);
        controls.dense_cb->setChecked(true);
        controls.dense_cb->blockSignals(false);
    }

    if (controls.sparse_mode_cb &&
        controls.sparse_mode_cb->currentIndex() != kSparseDa3) {
        BlockCombo(controls.sparse_mode_cb, kSparseDa3);
    }

    if (controls.da3_model_cb &&
        !IsNestedModelIndex(controls.da3_model_cb->currentIndex())) {
        BlockCombo(controls.da3_model_cb, kModelNestedAnyView);
        OnModelChanged(controls, kModelNestedAnyView);
    }
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
        return;
    }

    const int model_index =
        controls.da3_model_cb ? controls.da3_model_cb->currentIndex() : 0;
    SetStereoDa3ItemEnabled(controls.stereo_mode_cb,
                            IsNestedModelIndex(model_index));
}

void DA3ReconstructionUiBindings::Sync(
    const DA3ReconstructionUiControls& controls) {
    const int model_index =
        controls.da3_model_cb ? controls.da3_model_cb->currentIndex() : 0;
    OnModelChanged(controls, model_index);
    OnDenseChanged(controls,
                   !controls.dense_cb || controls.dense_cb->isChecked());
}

void DA3ReconstructionUiBindings::Install(
    const DA3ReconstructionUiControls& controls, QObject* context) {
    if (!context) {
        return;
    }

    if (controls.da3_model_cb) {
        QObject::connect(
            controls.da3_model_cb,
            QOverload<int>::of(&QComboBox::currentIndexChanged), context,
            [controls](int index) { OnModelChanged(controls, index); });
    }

    if (controls.stereo_mode_cb) {
        QObject::connect(
            controls.stereo_mode_cb,
            QOverload<int>::of(&QComboBox::currentIndexChanged), context,
            [controls](int index) { OnStereoModeChanged(controls, index); });
    }

    if (controls.dense_cb) {
        QObject::connect(controls.dense_cb, &QCheckBox::toggled, context,
                         [controls](bool checked) {
                             OnDenseChanged(controls, checked);
                         });
    }

    Sync(controls);
}

}  // namespace colmap
