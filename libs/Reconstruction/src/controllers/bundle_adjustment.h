// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "controllers/option_manager.h"
#include "scene/reconstruction.h"
#include "util/threading.h"

namespace colmap {

// Class that controls the global bundle adjustment procedure.
class BundleAdjustmentController : public Thread {
public:
    BundleAdjustmentController(const OptionManager& options,
                               Reconstruction* reconstruction);

private:
    void Run();

    const OptionManager options_;
    Reconstruction* reconstruction_;
};

}  // namespace colmap
