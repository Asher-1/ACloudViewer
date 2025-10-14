// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BundleAdjustmentWidget.h"

#include "OptionManager.h"
#include "ReconstructionWidget.h"
#include "ThreadControlWidget.h"
#include "controllers/BundleAdjustmentController.h"

namespace cloudViewer {

using namespace colmap;
BundleAdjustmentWidget::BundleAdjustmentWidget(
        ReconstructionWidget* main_window, OptionManager* options)
    : OptionsWidget(main_window),
      main_window_(main_window),
      options_(options),
      reconstruction_(nullptr),
      thread_control_widget_(new ThreadControlWidget(this)) {
    setWindowTitle("Bundle adjustment");

    AddOptionInt(&options->bundle_adjustment->solver_options.max_num_iterations,
                 "max_num_iterations");
    AddOptionInt(&options->bundle_adjustment->solver_options
                          .max_linear_solver_iterations,
                 "max_linear_solver_iterations");

    AddOptionDoubleLog(
            &options->bundle_adjustment->solver_options.function_tolerance,
            "function_tolerance [10eX]", -1000, 1000);
    AddOptionDoubleLog(
            &options->bundle_adjustment->solver_options.gradient_tolerance,
            "gradient_tolerance [10eX]", -1000, 1000);
    AddOptionDoubleLog(
            &options->bundle_adjustment->solver_options.parameter_tolerance,
            "parameter_tolerance [10eX]", -1000, 1000);

    AddOptionBool(&options->bundle_adjustment->refine_focal_length,
                  "refine_focal_length");
    AddOptionBool(&options->bundle_adjustment->refine_principal_point,
                  "refine_principal_point");
    AddOptionBool(&options->bundle_adjustment->refine_extra_params,
                  "refine_extra_params");
    AddOptionBool(&options->bundle_adjustment->refine_extrinsics,
                  "refine_extrinsics");

    QPushButton* run_button = new QPushButton(tr("Run"), this);
    grid_layout_->addWidget(run_button, grid_layout_->rowCount(), 1);
    connect(run_button, &QPushButton::released, this,
            &BundleAdjustmentWidget::Run);

    render_action_ = new QAction(this);
    connect(render_action_, &QAction::triggered, this,
            &BundleAdjustmentWidget::Render, Qt::QueuedConnection);
}

void BundleAdjustmentWidget::Show(Reconstruction* reconstruction) {
    reconstruction_ = reconstruction;
    show();
    raise();
}

void BundleAdjustmentWidget::Run() {
    CHECK_NOTNULL(reconstruction_);

    WriteOptions();

    Thread* thread = new BundleAdjustmentController(*options_, reconstruction_);
    thread->AddCallback(Thread::FINISHED_CALLBACK,
                        [this]() { render_action_->trigger(); });

    // Normalize scene for numerical stability and
    // to avoid large scale changes in viewer.
    reconstruction_->Normalize();

    thread_control_widget_->StartThread("Bundle adjusting...", true, thread);
}

void BundleAdjustmentWidget::Render() { main_window_->RenderNow(); }

}  // namespace cloudViewer
