// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LasCommands.h"

#include <QObject>

static const char COMMAND_LAS[] = "LAS";
static const char COMMAND_LAS_EXTRA_FIELDS[] = "EXTRA_FIELDS";
static const char COMMAND_LAS_TILE_SIZE[] = "TILE_SIZE";
static const char COMMAND_LAS_SAVE_LAZ[] = "SAVE_LAZ";
static const char COMMAND_LAS_VERSION[] = "LAS_VERSION";

CommandLAS::CommandLAS()
        : ccCommandLineInterface::Command("LAS", COMMAND_LAS) {}

bool CommandLAS::process(ccCommandLineInterface& cmd) {
    cmd.print("[LAS]");

    bool extraFields = false;
    double tileSize = 0.0;
    bool saveLaz = false;
    QString lasVersion;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_LAS_EXTRA_FIELDS)) {
            cmd.arguments().pop_front();
            extraFields = true;
            cmd.print("[LAS] Extra fields enabled");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_LAS_TILE_SIZE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_LAS_TILE_SIZE));
            bool ok;
            tileSize = cmd.arguments().takeFirst().toDouble(&ok);
            if (!ok || tileSize <= 0)
                return cmd.error("Invalid value for -TILE_SIZE");
            cmd.print(QObject::tr("[LAS] Tile size: %1").arg(tileSize));
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_LAS_SAVE_LAZ)) {
            cmd.arguments().pop_front();
            saveLaz = true;
            cmd.print("[LAS] Save as LAZ enabled");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_LAS_VERSION)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_LAS_VERSION));
            lasVersion = cmd.arguments().takeFirst();
            cmd.print(QObject::tr("[LAS] Version: %1").arg(lasVersion));
        } else {
            break;
        }
    }

    Q_UNUSED(extraFields);
    Q_UNUSED(tileSize);
    Q_UNUSED(saveLaz);
    return true;
}
