// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvCommandLineInterface.h"

class ecvProgressDialog;
class ecvMainAppInterface;

#include <ecvHObject.h>

class PCVCommand : public ccCommandLineInterface::Command {
public:
    PCVCommand();

    ~PCVCommand() override = default;

    static bool Process(const ccHObject::Container& candidates,
                        const std::vector<CCVector3d>& rays,
                        bool meshIsClosed,
                        unsigned resolution,
                        ecvProgressDialog* progressDlg = nullptr,
                        ecvMainAppInterface* app = nullptr);

    bool process(ccCommandLineInterface& cmd) override;
};
