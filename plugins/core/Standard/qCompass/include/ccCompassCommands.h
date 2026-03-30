// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvCommandLineInterface.h"

struct CommandCompassExport : public ccCommandLineInterface::Command {
    CommandCompassExport();
    bool process(ccCommandLineInterface& cmd) override;
};

struct CommandCompassImportFoliations : public ccCommandLineInterface::Command {
    CommandCompassImportFoliations();
    bool process(ccCommandLineInterface& cmd) override;
};

struct CommandCompassImportLineations : public ccCommandLineInterface::Command {
    CommandCompassImportLineations();
    bool process(ccCommandLineInterface& cmd) override;
};

struct CommandCompassRefit : public ccCommandLineInterface::Command {
    CommandCompassRefit();
    bool process(ccCommandLineInterface& cmd) override;
};

struct CommandCompassP21 : public ccCommandLineInterface::Command {
    CommandCompassP21();
    bool process(ccCommandLineInterface& cmd) override;
};
