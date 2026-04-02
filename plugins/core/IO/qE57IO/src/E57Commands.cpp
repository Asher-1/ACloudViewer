// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "E57Commands.h"

#include <QObject>

static const char COMMAND_E57[] = "E57";
static const char COMMAND_E57_GLOBAL_SHIFT[] = "GLOBAL_SHIFT";
static const char COMMAND_E57_IGNORE_INTENSITY[] = "IGNORE_INTENSITY";
static const char COMMAND_E57_IGNORE_COLOR[] = "IGNORE_COLOR";

CommandE57::CommandE57()
    : ccCommandLineInterface::Command("E57", COMMAND_E57) {}

bool CommandE57::process(ccCommandLineInterface& cmd) {
    cmd.print("[E57]");

    bool ignoreIntensity = false;
    bool ignoreColor = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_E57_GLOBAL_SHIFT)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_E57_GLOBAL_SHIFT));
            QString shift = cmd.arguments().takeFirst();
            cmd.print(QObject::tr("[E57] Global shift: %1").arg(shift));
        } else if (ccCommandLineInterface::IsCommand(
                           arg, COMMAND_E57_IGNORE_INTENSITY)) {
            cmd.arguments().pop_front();
            ignoreIntensity = true;
            cmd.print("[E57] Ignore intensity enabled");
        } else if (ccCommandLineInterface::IsCommand(
                           arg, COMMAND_E57_IGNORE_COLOR)) {
            cmd.arguments().pop_front();
            ignoreColor = true;
            cmd.print("[E57] Ignore color enabled");
        } else {
            break;
        }
    }

    Q_UNUSED(ignoreIntensity);
    Q_UNUSED(ignoreColor);
    return true;
}
