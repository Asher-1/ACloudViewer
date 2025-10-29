// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FBXCommand.h"

#include "FBXFilter.h"

constexpr char COMMAND_FBX[] = "FBX";
constexpr char COMMAND_FBX_EXPORT_FORMAT[] = "EXPORT_FMT";

FBXCommand::FBXCommand() : Command("FBX", COMMAND_FBX) {}

bool FBXCommand::process(ccCommandLineInterface& cmd) {
    cmd.print("[FBX]");

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();

        if (ccCommandLineInterface::IsCommand(arg, COMMAND_FBX_EXPORT_FORMAT)) {
            cmd.arguments().pop_front();

            QString format = cmd.arguments().takeFirst();

            if (format.isNull()) {
                return cmd.error(QObject::tr("Missing parameter: FBX format "
                                             "(string) after '%1'")
                                         .arg(COMMAND_FBX_EXPORT_FORMAT));
            }

            cmd.print(QObject::tr("FBX format: %1").arg(format));

            FBXFilter::SetDefaultOutputFormat(format);
        }
    }

    return true;
}
