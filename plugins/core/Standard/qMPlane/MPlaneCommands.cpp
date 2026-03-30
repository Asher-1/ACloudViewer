// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MPlaneCommands.h"

#include <ecvPointCloud.h>

#include <QObject>

static const char COMMAND_MPLANE[] = "MPLANE";
static const char COMMAND_MP_NX[] = "NX";
static const char COMMAND_MP_NY[] = "NY";
static const char COMMAND_MP_NZ[] = "NZ";
static const char COMMAND_MP_D[] = "D";

CommandMPlane::CommandMPlane()
        : ccCommandLineInterface::Command("MPlane", COMMAND_MPLANE) {}

bool CommandMPlane::process(ccCommandLineInterface& cmd) {
    cmd.print("[MPLANE]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_MPLANE));
    }

    double nx = 0, ny = 0, nz = 1, d = 0;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        bool ok = false;
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_MP_NX)) {
            cmd.arguments().pop_front();
            nx = cmd.arguments().takeFirst().toDouble(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_MP_NY)) {
            cmd.arguments().pop_front();
            ny = cmd.arguments().takeFirst().toDouble(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_MP_NZ)) {
            cmd.arguments().pop_front();
            nz = cmd.arguments().takeFirst().toDouble(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_MP_D)) {
            cmd.arguments().pop_front();
            d = cmd.arguments().takeFirst().toDouble(&ok);
        } else {
            break;
        }
        if (!ok) return cmd.error("Invalid MPlane parameter value");
    }

    double len = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (len < 1e-12) {
        return cmd.error("[MPLANE] Normal vector must be non-zero");
    }
    nx /= len;
    ny /= len;
    nz /= len;

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* pc = desc.pc;
        if (!pc) continue;

        cmd.print(QObject::tr("[MPLANE] Computing plane distance for cloud "
                              "'%1' (%2 points)")
                          .arg(pc->getName())
                          .arg(pc->size()));
        cmd.print(QObject::tr("[MPLANE] Plane normal: (%1, %2, %3), d: %4")
                          .arg(nx).arg(ny).arg(nz).arg(d));

        int sfIdx = pc->getScalarFieldIndexByName("Plane Distance");
        if (sfIdx < 0) {
            sfIdx = pc->addScalarField("Plane Distance");
        }
        if (sfIdx < 0) {
            return cmd.error("[MPLANE] Failed to create scalar field");
        }

        cloudViewer::ScalarField* sf = pc->getScalarField(sfIdx);
        for (unsigned i = 0; i < pc->size(); ++i) {
            const CCVector3* P = pc->getPoint(i);
            double dist = nx * P->x + ny * P->y + nz * P->z + d;
            sf->setValue(i, static_cast<ScalarType>(dist));
        }
        sf->computeMinAndMax();
        pc->setCurrentDisplayedScalarField(sfIdx);
        pc->showSF(true);

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_MPLANE");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}
