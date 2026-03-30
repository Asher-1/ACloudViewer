// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qSRACommands.h"

#include "distanceMapGenerationTool.h"
#include "profileLoader.h"

#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>

#include <QObject>

static const char COMMAND_SRA[] = "SRA";
static const char COMMAND_SRA_PROFILE[] = "PROFILE";
static const char COMMAND_SRA_AXIS[] = "AXIS";

CommandSRARadialDist::CommandSRARadialDist()
        : ccCommandLineInterface::Command("SRA Radial Distance", COMMAND_SRA) {
}

bool CommandSRARadialDist::process(ccCommandLineInterface& cmd) {
    cmd.print("[SRA]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_SRA));
    }

    QString profileFile;
    unsigned char revolAxis = 2;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_SRA_PROFILE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_SRA_PROFILE));
            profileFile = cmd.arguments().takeFirst();
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_SRA_AXIS)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_SRA_AXIS));
            QString axisStr = cmd.arguments().takeFirst().toUpper();
            if (axisStr == "X")
                revolAxis = 0;
            else if (axisStr == "Y")
                revolAxis = 1;
            else if (axisStr == "Z")
                revolAxis = 2;
            else
                return cmd.error(QObject::tr("Invalid axis '%1'. Use X, Y, "
                                             "or Z.")
                                         .arg(axisStr));
        } else {
            break;
        }
    }

    if (profileFile.isEmpty()) {
        return cmd.error(
                QObject::tr("Missing profile file (use \"-%1 <path>\")")
                        .arg(COMMAND_SRA_PROFILE));
    }

    CCVector3 origin(0, 0, 0);
    ccPolyline* profile = ProfileLoader::Load(profileFile, origin, nullptr);
    if (!profile) {
        return cmd.error(
                QObject::tr("Failed to load profile from '%1'")
                        .arg(profileFile));
    }

    DistanceMapGenerationTool::SetPoylineOrigin(profile, origin);
    DistanceMapGenerationTool::SetPoylineRevolDim(profile, revolAxis);

    size_t errorCount = 0;
    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* cloud = desc.pc;
        if (!cloud) {
            ++errorCount;
            continue;
        }

        cmd.print(QObject::tr("[SRA] Computing radial distance for cloud '%1'")
                          .arg(cloud->getName()));

        if (!DistanceMapGenerationTool::ComputeRadialDist(cloud, profile, true,
                                                          nullptr)) {
            cmd.warning(QObject::tr("[SRA] Failed for cloud '%1'")
                                .arg(cloud->getName()));
            ++errorCount;
            continue;
        }

        int sfIdx = cloud->getScalarFieldIndexByName(RADIAL_DIST_SF_NAME);
        if (sfIdx >= 0) {
            ccScalarField* sf = static_cast<ccScalarField*>(
                    cloud->getScalarField(sfIdx));
            if (sf) {
                sf->computeMinAndMax();
                cloud->setCurrentDisplayedScalarField(sfIdx);
                cloud->showSF(true);
            }
        }

        desc.basename += QString("_SRA");
        if (cmd.autoSaveMode()) {
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) {
                delete profile;
                return cmd.error(errorStr);
            }
        }
    }

    delete profile;

    if (errorCount > 0) {
        return cmd.error(QObject::tr("[SRA] %1 error(s) occurred")
                                 .arg(errorCount));
    }

    return true;
}
