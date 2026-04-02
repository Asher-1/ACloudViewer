// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MeshIOCommands.h"

#include <QObject>

static const char COMMAND_MESH_IO[] = "MESH_IO";
static const char COMMAND_MIO_SCALE[] = "SCALE";
static const char COMMAND_MIO_UP_AXIS[] = "UP_AXIS";
static const char COMMAND_MIO_MERGE_NODES[] = "MERGE_NODES";

CommandMeshIO::CommandMeshIO()
    : ccCommandLineInterface::Command("Mesh IO", COMMAND_MESH_IO) {}

bool CommandMeshIO::process(ccCommandLineInterface& cmd) {
    cmd.print("[MESH_IO]");

    float scale = 1.0f;
    QString upAxis = "Y";
    bool mergeNodes = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_MIO_SCALE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_MIO_SCALE));
            bool ok;
            scale = cmd.arguments().takeFirst().toFloat(&ok);
            if (!ok || scale <= 0) return cmd.error("Invalid value for -SCALE");
            cmd.print(QObject::tr("[MESH_IO] Scale: %1").arg(scale));
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_MIO_UP_AXIS)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_MIO_UP_AXIS));
            upAxis = cmd.arguments().takeFirst().toUpper();
            if (upAxis != "X" && upAxis != "Y" && upAxis != "Z")
                return cmd.error("Invalid -UP_AXIS (use X, Y, or Z)");
            cmd.print(QObject::tr("[MESH_IO] Up axis: %1").arg(upAxis));
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_MIO_MERGE_NODES)) {
            cmd.arguments().pop_front();
            mergeNodes = true;
            cmd.print("[MESH_IO] Merge nodes enabled");
        } else {
            break;
        }
    }

    Q_UNUSED(scale);
    Q_UNUSED(mergeNodes);
    return true;
}
