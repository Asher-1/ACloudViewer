// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CoreIOCommands.h"

#include <QObject>

static const char COMMAND_CORE_IO[] = "CORE_IO";
static const char COMMAND_CIO_FORMAT[] = "FORMAT";
static const char COMMAND_CIO_PRECISION[] = "PRECISION";

CommandCoreIO::CommandCoreIO()
        : ccCommandLineInterface::Command("Core IO", COMMAND_CORE_IO) {}

bool CommandCoreIO::process(ccCommandLineInterface& cmd) {
    cmd.print("[CORE_IO]");

    QString format;
    int precision = -1;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CIO_FORMAT)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CIO_FORMAT));
            format = cmd.arguments().takeFirst();
            cmd.print(QObject::tr("[CORE_IO] Format: %1").arg(format));
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_CIO_PRECISION)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CIO_PRECISION));
            bool ok;
            precision = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || precision < 0)
                return cmd.error("Invalid value for -PRECISION");
            cmd.print(QObject::tr("[CORE_IO] Precision: %1").arg(precision));
        } else {
            break;
        }
    }

    return true;
}
