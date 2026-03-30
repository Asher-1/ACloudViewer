// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qVoxFallCommands.h"

#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>

#include <QObject>
#include <QSettings>

#include "qVoxFallDialog.h"
#include "qVoxFallProcess.h"

static const char COMMAND_VOXFALL[] = "VOXFALL";
static const char COMMAND_VF_VOXEL_SIZE[] = "VOXEL_SIZE";
static const char COMMAND_VF_AZIMUTH[] = "AZIMUTH";
static const char COMMAND_VF_EXPORT_MESHES[] = "EXPORT_MESHES";
static const char COMMAND_VF_LOSS_GAIN[] = "LOSS_GAIN";

CommandVoxFall::CommandVoxFall()
    : ccCommandLineInterface::Command("VoxFall", COMMAND_VOXFALL) {}

bool CommandVoxFall::process(ccCommandLineInterface& cmd) {
    cmd.print("[VOXFALL]");

    if (cmd.meshes().size() < 2) {
        return cmd.error(QObject::tr("At least two meshes are required (use "
                                     "\"-O\" before \"-%1\")")
                                 .arg(COMMAND_VOXFALL));
    }

    double voxelSize = 0.1;
    double azimuth = 0.0;
    bool exportMeshes = false;
    bool lossGain = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_VF_VOXEL_SIZE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty()) {
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_VF_VOXEL_SIZE));
            }
            bool ok = false;
            voxelSize = cmd.arguments().takeFirst().toDouble(&ok);
            if (!ok || voxelSize <= 0.0) {
                return cmd.error(QObject::tr("Invalid value for \"-%1\"")
                                         .arg(COMMAND_VF_VOXEL_SIZE));
            }
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_VF_AZIMUTH)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty()) {
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_VF_AZIMUTH));
            }
            bool ok = false;
            azimuth = cmd.arguments().takeFirst().toDouble(&ok);
            if (!ok) {
                return cmd.error(QObject::tr("Invalid value for \"-%1\"")
                                         .arg(COMMAND_VF_AZIMUTH));
            }
        } else if (ccCommandLineInterface::IsCommand(
                           arg, COMMAND_VF_EXPORT_MESHES)) {
            cmd.arguments().pop_front();
            exportMeshes = true;
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_VF_LOSS_GAIN)) {
            cmd.arguments().pop_front();
            lossGain = true;
        } else {
            break;
        }
    }

    ccMesh* mesh1 = ccHObjectCaster::ToMesh(cmd.meshes()[0].mesh);
    ccMesh* mesh2 = ccHObjectCaster::ToMesh(cmd.meshes()[1].mesh);
    if (!mesh1 || !mesh2) {
        return cmd.error(QObject::tr("Invalid mesh entity"));
    }

    qVoxFallDialog dlg(mesh1, mesh2, nullptr);
    QSettings mem;
    mem.setValue("VoxelSize", voxelSize);
    mem.setValue("Azimuth", azimuth);
    mem.setValue("ExportMeshesEnabled", exportMeshes);
    mem.setValue("LossGainEnabled", lossGain);
    dlg.loadParamsFrom(mem);

    ccPointCloud* outGrid = nullptr;
    ccHObject* outClusters = nullptr;
    QString errorMessage;

    ccPointCloud** pGrid = &outGrid;
    ccHObject** pClusters = exportMeshes ? &outClusters : nullptr;

    if (!qVoxFallProcess::Compute(dlg, errorMessage, false, cmd.widgetParent(),
                                  nullptr, pGrid, pClusters)) {
        return cmd.error(errorMessage.isEmpty()
                                 ? QObject::tr("VoxFall computation failed")
                                 : errorMessage);
    }

    const QString base = cmd.meshes()[0].basename;

    if (outGrid) {
        CLCloudDesc cloudDesc(outGrid, base + "_VOXFALL", cmd.meshes()[0].path);
        cmd.clouds().push_back(cloudDesc);
        if (cmd.autoSaveMode()) {
            QString errStr = cmd.exportEntity(cmd.clouds().back());
            if (!errStr.isEmpty()) {
                return cmd.error(errStr);
            }
        }
    }

    if (outClusters) {
        if (cmd.autoSaveMode()) {
            CLGroupDesc grpDesc(outClusters, base + "_VOXFALL_clusters",
                                cmd.meshes()[0].path);
            QString errStr = cmd.exportEntity(
                    grpDesc, QString(), nullptr,
                    ccCommandLineInterface::ExportOption::ForceHierarchy);
            if (!errStr.isEmpty()) {
                return cmd.error(errStr);
            }
        }
        delete outClusters;
    }

    return true;
}
