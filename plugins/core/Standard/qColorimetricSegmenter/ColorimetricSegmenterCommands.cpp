// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ColorimetricSegmenterCommands.h"

#include <ecvPointCloud.h>

#include <QObject>

// --- RGB Filter Command ---

static const char COMMAND_COLOR_SEG_RGB[] = "COLOR_SEG_RGB";
static const char COMMAND_CS_R_MIN[] = "R_MIN";
static const char COMMAND_CS_R_MAX[] = "R_MAX";
static const char COMMAND_CS_G_MIN[] = "G_MIN";
static const char COMMAND_CS_G_MAX[] = "G_MAX";
static const char COMMAND_CS_B_MIN[] = "B_MIN";
static const char COMMAND_CS_B_MAX[] = "B_MAX";

CommandColorimetricSegRGB::CommandColorimetricSegRGB()
        : ccCommandLineInterface::Command("Colorimetric Seg RGB",
                                          COMMAND_COLOR_SEG_RGB) {}

bool CommandColorimetricSegRGB::process(ccCommandLineInterface& cmd) {
    cmd.print("[COLOR_SEG_RGB]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_COLOR_SEG_RGB));
    }

    int rMin = 0, rMax = 255, gMin = 0, gMax = 255, bMin = 0, bMax = 255;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        bool ok = false;
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_R_MIN)) {
            cmd.arguments().pop_front();
            rMin = cmd.arguments().takeFirst().toInt(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_R_MAX)) {
            cmd.arguments().pop_front();
            rMax = cmd.arguments().takeFirst().toInt(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_G_MIN)) {
            cmd.arguments().pop_front();
            gMin = cmd.arguments().takeFirst().toInt(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_G_MAX)) {
            cmd.arguments().pop_front();
            gMax = cmd.arguments().takeFirst().toInt(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_B_MIN)) {
            cmd.arguments().pop_front();
            bMin = cmd.arguments().takeFirst().toInt(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_B_MAX)) {
            cmd.arguments().pop_front();
            bMax = cmd.arguments().takeFirst().toInt(&ok);
        } else {
            break;
        }
        if (!ok) return cmd.error("Invalid RGB filter parameter value");
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* pc = desc.pc;
        if (!pc || !pc->hasColors()) {
            cmd.warning(QObject::tr("[COLOR_SEG_RGB] Cloud '%1' has no colors, "
                                    "skipping")
                                .arg(pc ? pc->getName() : "null"));
            continue;
        }

        cmd.print(QObject::tr("[COLOR_SEG_RGB] Filtering cloud '%1' with "
                              "R[%2,%3] G[%4,%5] B[%6,%7]")
                          .arg(pc->getName())
                          .arg(rMin).arg(rMax)
                          .arg(gMin).arg(gMax)
                          .arg(bMin).arg(bMax));

        unsigned count = pc->size();
        unsigned filtered = 0;
        for (unsigned i = 0; i < count; ++i) {
            const ecvColor::Rgb& color = pc->getPointColor(i);
            if (color.r >= rMin && color.r <= rMax &&
                color.g >= gMin && color.g <= gMax &&
                color.b >= bMin && color.b <= bMax) {
                ++filtered;
            }
        }

        cmd.print(QObject::tr("[COLOR_SEG_RGB] %1 of %2 points match filter "
                              "in cloud '%3'")
                          .arg(filtered).arg(count).arg(pc->getName()));

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_COLOR_SEG_RGB");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}

// --- HSV Filter Command ---

static const char COMMAND_COLOR_SEG_HSV[] = "COLOR_SEG_HSV";
static const char COMMAND_CS_H_MIN[] = "H_MIN";
static const char COMMAND_CS_H_MAX[] = "H_MAX";
static const char COMMAND_CS_S_MIN[] = "S_MIN";
static const char COMMAND_CS_S_MAX[] = "S_MAX";
static const char COMMAND_CS_V_MIN[] = "V_MIN";
static const char COMMAND_CS_V_MAX[] = "V_MAX";

CommandColorimetricSegHSV::CommandColorimetricSegHSV()
        : ccCommandLineInterface::Command("Colorimetric Seg HSV",
                                          COMMAND_COLOR_SEG_HSV) {}

bool CommandColorimetricSegHSV::process(ccCommandLineInterface& cmd) {
    cmd.print("[COLOR_SEG_HSV]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_COLOR_SEG_HSV));
    }

    float hMin = 0, hMax = 360, sMin = 0, sMax = 100, vMin = 0, vMax = 100;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        bool ok = false;
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_H_MIN)) {
            cmd.arguments().pop_front();
            hMin = cmd.arguments().takeFirst().toFloat(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_H_MAX)) {
            cmd.arguments().pop_front();
            hMax = cmd.arguments().takeFirst().toFloat(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_S_MIN)) {
            cmd.arguments().pop_front();
            sMin = cmd.arguments().takeFirst().toFloat(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_S_MAX)) {
            cmd.arguments().pop_front();
            sMax = cmd.arguments().takeFirst().toFloat(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_V_MIN)) {
            cmd.arguments().pop_front();
            vMin = cmd.arguments().takeFirst().toFloat(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_V_MAX)) {
            cmd.arguments().pop_front();
            vMax = cmd.arguments().takeFirst().toFloat(&ok);
        } else {
            break;
        }
        if (!ok) return cmd.error("Invalid HSV filter parameter value");
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* pc = desc.pc;
        if (!pc || !pc->hasColors()) {
            cmd.warning(QObject::tr("[COLOR_SEG_HSV] Cloud '%1' has no colors, "
                                    "skipping")
                                .arg(pc ? pc->getName() : "null"));
            continue;
        }

        cmd.print(QObject::tr("[COLOR_SEG_HSV] Filtering cloud '%1' with "
                              "H[%2,%3] S[%4,%5] V[%6,%7]")
                          .arg(pc->getName())
                          .arg(hMin).arg(hMax)
                          .arg(sMin).arg(sMax)
                          .arg(vMin).arg(vMax));

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_COLOR_SEG_HSV");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}

// --- Scalar Filter Command ---

static const char COMMAND_COLOR_SEG_SCALAR[] = "COLOR_SEG_SCALAR";
static const char COMMAND_CS_SCALAR_MIN[] = "SCALAR_MIN";
static const char COMMAND_CS_SCALAR_MAX[] = "SCALAR_MAX";

CommandColorimetricSegScalar::CommandColorimetricSegScalar()
        : ccCommandLineInterface::Command("Colorimetric Seg Scalar",
                                          COMMAND_COLOR_SEG_SCALAR) {}

bool CommandColorimetricSegScalar::process(ccCommandLineInterface& cmd) {
    cmd.print("[COLOR_SEG_SCALAR]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_COLOR_SEG_SCALAR));
    }

    double scalarMin = -1e38, scalarMax = 1e38;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        bool ok = false;
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CS_SCALAR_MIN)) {
            cmd.arguments().pop_front();
            scalarMin = cmd.arguments().takeFirst().toDouble(&ok);
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_CS_SCALAR_MAX)) {
            cmd.arguments().pop_front();
            scalarMax = cmd.arguments().takeFirst().toDouble(&ok);
        } else {
            break;
        }
        if (!ok) return cmd.error("Invalid scalar filter parameter value");
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* pc = desc.pc;
        if (!pc) continue;

        if (!pc->hasScalarFields()) {
            cmd.warning(QObject::tr("[COLOR_SEG_SCALAR] Cloud '%1' has no "
                                    "scalar fields, skipping")
                                .arg(pc->getName()));
            continue;
        }

        cmd.print(QObject::tr("[COLOR_SEG_SCALAR] Filtering cloud '%1' with "
                              "scalar range [%2, %3]")
                          .arg(pc->getName())
                          .arg(scalarMin)
                          .arg(scalarMax));

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_COLOR_SEG_SCALAR");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}
