// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ecvCommandLineInterface.h"

class QString;
class QXmlStreamAttributes;

struct CommandCrossSection : public ccCommandLineInterface::Command {
    CommandCrossSection();

    bool process(ccCommandLineInterface& cmd) override;

private:
    bool readVector(const QXmlStreamAttributes& attributes,
                    CCVector3& P,
                    QString element,
                    const ccCommandLineInterface& cmd);
};
