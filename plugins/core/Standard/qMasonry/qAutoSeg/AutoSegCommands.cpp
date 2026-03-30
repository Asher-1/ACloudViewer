// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AutoSegCommands.h"

#include <ecvPointCloud.h>

#include <QObject>

static const char COMMAND_AUTO_SEG[] = "AUTO_SEG";
static const char COMMAND_AS_MORTAR_MAPS[] = "MORTAR_MAPS";
static const char COMMAND_AS_CONTOURS[] = "CONTOURS";
static const char COMMAND_AS_PROFILE[] = "PROFILE";

CommandAutoSeg::CommandAutoSeg()
        : ccCommandLineInterface::Command("AutoSeg", COMMAND_AUTO_SEG) {}

bool CommandAutoSeg::process(ccCommandLineInterface& cmd) {
    cmd.print("[AUTO_SEG]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_AUTO_SEG));
    }

    bool mortarMaps = false;
    bool contours = false;
    QString profileFile;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_AS_MORTAR_MAPS)) {
            cmd.arguments().pop_front();
            mortarMaps = true;
            cmd.print("[AUTO_SEG] Mortar maps enabled");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_AS_CONTOURS)) {
            cmd.arguments().pop_front();
            contours = true;
            cmd.print("[AUTO_SEG] Contour extraction enabled");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_AS_PROFILE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_AS_PROFILE));
            profileFile = cmd.arguments().takeFirst();
            cmd.print(QObject::tr("[AUTO_SEG] Profile file: %1")
                              .arg(profileFile));
        } else {
            break;
        }
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* pc = desc.pc;
        if (!pc) continue;

        cmd.print(QObject::tr("[AUTO_SEG] Processing cloud '%1' (%2 points)")
                          .arg(pc->getName())
                          .arg(pc->size()));

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_AUTO_SEG");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}
