// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvCommandLineInterface.h"

//! CLI: mesh boolean (Cork CSG)
class CommandCork : public ccCommandLineInterface::Command {
public:
    CommandCork();
    bool process(ccCommandLineInterface& cmd) override;
};
