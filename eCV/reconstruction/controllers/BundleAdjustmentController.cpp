// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BundleAdjustmentController.h"

#include <ceres/ceres.h>

#include "base/reconstruction.h"
#include "optim/bundle_adjustment.h"
#include "util/misc.h"

namespace cloudViewer {
namespace {

// Callback functor called after each bundle adjustment iteration.
class BundleAdjustmentIterationCallback : public ceres::IterationCallback {
public:
    explicit BundleAdjustmentIterationCallback(colmap::Thread* thread)
        : thread_(thread) {}

    virtual ceres::CallbackReturnType operator()(
            const ceres::IterationSummary& summary) {
        CHECK_NOTNULL(thread_);
        thread_->BlockIfPaused();
        if (thread_->IsStopped()) {
            return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
        } else {
            return ceres::SOLVER_CONTINUE;
        }
    }

private:
    colmap::Thread* thread_;
};

}  // namespace

using namespace colmap;

BundleAdjustmentController::BundleAdjustmentController(
        const OptionManager& options, Reconstruction* reconstruction)
    : options_(options), reconstruction_(reconstruction) {}

void BundleAdjustmentController::Run() {
    CHECK_NOTNULL(reconstruction_);

    PrintHeading1("Global bundle adjustment");

    const std::vector<image_t>& reg_image_ids = reconstruction_->RegImageIds();

    if (reg_image_ids.size() < 2) {
        std::cout << "ERROR: Need at least two views." << std::endl;
        return;
    }

    // Avoid degeneracies in bundle adjustment.
    reconstruction_->FilterObservationsWithNegativeDepth();

    BundleAdjustmentOptions ba_options = *options_.bundle_adjustment;
    ba_options.solver_options.minimizer_progress_to_stdout = true;

    BundleAdjustmentIterationCallback iteration_callback(this);
    ba_options.solver_options.callbacks.push_back(&iteration_callback);

    // Configure bundle adjustment.
    BundleAdjustmentConfig ba_config;
    for (const image_t image_id : reg_image_ids) {
        ba_config.AddImage(image_id);
    }
    ba_config.SetConstantPose(reg_image_ids[0]);
    ba_config.SetConstantTvec(reg_image_ids[1], {0});

    // Run bundle adjustment.
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    bundle_adjuster.Solve(reconstruction_);

    GetTimer().PrintMinutes();
}

}  // namespace cloudViewer
