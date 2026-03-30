// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvCommandLineInterface.h"

struct CommandColorimetricSegRGB : public ccCommandLineInterface::Command {
    CommandColorimetricSegRGB();
    bool process(ccCommandLineInterface& cmd) override;
};

struct CommandColorimetricSegHSV : public ccCommandLineInterface::Command {
    CommandColorimetricSegHSV();
    bool process(ccCommandLineInterface& cmd) override;
};

struct CommandColorimetricSegScalar : public ccCommandLineInterface::Command {
    CommandColorimetricSegScalar();
    bool process(ccCommandLineInterface& cmd) override;
};
