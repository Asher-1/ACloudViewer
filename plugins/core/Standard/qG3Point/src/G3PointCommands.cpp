// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "G3PointCommands.h"

#include <ecvPointCloud.h>

#include <QObject>

static const char COMMAND_G3POINT[] = "G3POINT";
static const char COMMAND_G3P_MAX_RADIUS[] = "MAX_RADIUS";
static const char COMMAND_G3P_MIN_RADIUS[] = "MIN_RADIUS";
static const char COMMAND_G3P_N_NEIGHBORS[] = "N_NEIGHBORS";
static const char COMMAND_G3P_EXPORT_ELLIPSOIDS[] = "EXPORT_ELLIPSOIDS";

CommandG3Point::CommandG3Point()
    : ccCommandLineInterface::Command("G3Point", COMMAND_G3POINT) {}

bool CommandG3Point::process(ccCommandLineInterface& cmd) {
    cmd.print("[G3POINT]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr("No point cloud loaded (use \"-O "
                                     "[filename]\" before \"-%1\")")
                                 .arg(COMMAND_G3POINT));
    }

    double maxRadius = 0.0;
    double minRadius = 0.0;
    int nNeighbors = 30;
    bool exportEllipsoids = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_G3P_MAX_RADIUS)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_G3P_MAX_RADIUS));
            bool ok;
            maxRadius = cmd.arguments().takeFirst().toDouble(&ok);
            if (!ok || maxRadius < 0)
                return cmd.error("Invalid value for -MAX_RADIUS");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_G3P_MIN_RADIUS)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_G3P_MIN_RADIUS));
            bool ok;
            minRadius = cmd.arguments().takeFirst().toDouble(&ok);
            if (!ok || minRadius < 0)
                return cmd.error("Invalid value for -MIN_RADIUS");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_G3P_N_NEIGHBORS)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_G3P_N_NEIGHBORS));
            bool ok;
            nNeighbors = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok || nNeighbors < 1)
                return cmd.error("Invalid value for -N_NEIGHBORS");
        } else if (ccCommandLineInterface::IsCommand(
                           arg, COMMAND_G3P_EXPORT_ELLIPSOIDS)) {
            cmd.arguments().pop_front();
            exportEllipsoids = true;
        } else {
            break;
        }
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* pc = desc.pc;
        if (!pc) continue;

        cmd.print(QObject::tr("[G3POINT] Processing cloud '%1' (%2 points)")
                          .arg(pc->getName())
                          .arg(pc->size()));
        cmd.print(QObject::tr("[G3POINT] Parameters - neighbors: %1, "
                              "min radius: %2, max radius: %3")
                          .arg(nNeighbors)
                          .arg(minRadius)
                          .arg(maxRadius));

        if (exportEllipsoids) {
            cmd.print("[G3POINT] Ellipsoid export enabled");
        }

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_G3POINT");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}
