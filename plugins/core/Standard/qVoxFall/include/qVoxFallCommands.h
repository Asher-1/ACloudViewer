// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvCommandLineInterface.h"

//! CLI: voxel rockfall detection (VoxFall)
class CommandVoxFall : public ccCommandLineInterface::Command {
public:
    CommandVoxFall();
    bool process(ccCommandLineInterface& cmd) override;
};
