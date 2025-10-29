// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "../OptionManager.h"
#include "util/threading.h"

namespace colmap {
class Reconstruction;
}

namespace cloudViewer {

// Class that controls the global bundle adjustment procedure.
class BundleAdjustmentController : public colmap::Thread {
public:
    BundleAdjustmentController(const OptionManager& options,
                               colmap::Reconstruction* reconstruction);

private:
    void Run();

    const OptionManager options_;
    colmap::Reconstruction* reconstruction_;
};

}  // namespace cloudViewer
