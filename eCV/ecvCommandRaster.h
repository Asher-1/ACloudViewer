// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvCommandLineInterface.h"

struct CommandRasterize : public ccCommandLineInterface::Command {
    CommandRasterize();

    bool process(ccCommandLineInterface& cmd) override;
};

struct CommandVolume25D : public ccCommandLineInterface::Command {
    CommandVolume25D();

    bool process(ccCommandLineInterface& cmd) override;
};
