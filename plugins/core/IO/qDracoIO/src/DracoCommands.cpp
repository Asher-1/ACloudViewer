// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DracoCommands.h"

#include <QObject>

static const char COMMAND_DRACO[] = "DRACO";
static const char COMMAND_DRC_QUANTIZATION[] = "QUANTIZATION";
static const char COMMAND_DRC_COMPRESSION[] = "COMPRESSION_LEVEL";
static const char COMMAND_DRC_SPEED[] = "SPEED";

CommandDraco::CommandDraco()
        : ccCommandLineInterface::Command("Draco", COMMAND_DRACO) {}

bool CommandDraco::process(ccCommandLineInterface& cmd) {
    cmd.print("[DRACO]");

    int quantization = 11;
    int compressionLevel = 7;
    int speed = 5;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_DRC_QUANTIZATION)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_DRC_QUANTIZATION));
            bool ok;
            quantization = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || quantization < 0 || quantization > 30)
                return cmd.error("Invalid value for -QUANTIZATION (0-30)");
            cmd.print(QObject::tr("[DRACO] Quantization bits: %1")
                              .arg(quantization));
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_DRC_COMPRESSION)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_DRC_COMPRESSION));
            bool ok;
            compressionLevel = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || compressionLevel < 0 || compressionLevel > 10)
                return cmd.error(
                        "Invalid value for -COMPRESSION_LEVEL (0-10)");
            cmd.print(QObject::tr("[DRACO] Compression level: %1")
                              .arg(compressionLevel));
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_DRC_SPEED)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_DRC_SPEED));
            bool ok;
            speed = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || speed < 0 || speed > 10)
                return cmd.error("Invalid value for -SPEED (0-10)");
            cmd.print(QObject::tr("[DRACO] Speed: %1").arg(speed));
        } else {
            break;
        }
    }

    Q_UNUSED(quantization);
    Q_UNUSED(compressionLevel);
    Q_UNUSED(speed);
    return true;
}
