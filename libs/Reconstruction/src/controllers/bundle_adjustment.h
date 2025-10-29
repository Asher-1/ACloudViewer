// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "base/reconstruction.h"
#include "util/option_manager.h"
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
