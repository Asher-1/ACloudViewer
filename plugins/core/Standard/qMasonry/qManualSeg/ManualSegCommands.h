// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvCommandLineInterface.h"

struct CommandManualSeg : public ccCommandLineInterface::Command {
    CommandManualSeg();
    bool process(ccCommandLineInterface& cmd) override;
};
