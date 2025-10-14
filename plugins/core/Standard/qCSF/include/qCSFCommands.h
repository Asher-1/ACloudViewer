// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QCSF_PLUGIN_COMMANDS_HEADER
#define QCSF_PLUGIN_COMMANDS_HEADER

// CloudViewer
#include "ecvCommandLineInterface.h"

// Local
#include "CSF.h"
#include "ccCSFDlg.h"

static const char COMMAND_CSF[] = "CSF";
static const char COMMAND_CSF_SCENE[] = "SCENES";
static const char COMMAND_CSF_SCENE_SLOPE[] = "SLOPE";
static const char COMMAND_CSF_SCENE_RELIEF[] = "RELIEF";
static const char COMMAND_CSF_SCENE_FLAT[] = "FLAT";
static const char COMMAND_CSF_PROC_SLOPE[] = "PROC_SLOPE";
static const char COMMAND_CSF_CLOTH_RESOLUTION[] = "CLOTH_RESOLUTION";
static const char COMMAND_CSF_MAX_ITERATION[] = "MAX_ITERATION";
static const char COMMAND_CSF_CLASS_THRESHOLD[] = "CLASS_THRESHOLD";
static const char COMMAND_CSF_EXPORT_GROUND[] = "EXPORT_GROUND";
static const char COMMAND_CSF_EXPORT_OFFGROUND[] = "EXPORT_OFFGROUND";

struct CommandCSF : public ccCommandLineInterface::Command {
    CommandCSF() : ccCommandLineInterface::Command("CSF", COMMAND_CSF) {}

    virtual bool process(ccCommandLineInterface& cmd) override {
        cmd.print("[CSF]");

        if (cmd.clouds().size() < 1) {
            cmd.error("No cloud loaded (1 is expected)");
            return false;
        }

        ccPointCloud* pc = cmd.clouds()[0].pc;

        // Convert CC point cloud to CSF type
        unsigned count = pc->size();
        wl::PointCloud csfPC;
        try {
            csfPC.reserve(count);
        } catch (const std::bad_alloc&) {
            return cmd.error("Not enough memory!");
        }

        for (unsigned i = 0; i < count; i++) {
            const CCVector3* P = pc->getPoint(i);
            wl::Point tmpPoint;
            tmpPoint.x = P->x;
            tmpPoint.y = -P->z;
            tmpPoint.z = P->y;
            csfPC.push_back(tmpPoint);
        }

        // initial parameters
        bool csfPostprocessing = false;
        double clothResolution = 2;
        double classThreshold = 0.5;
        int csfRigidness = 2;
        int maxIteration = 500;
        bool exportGround = false;
        bool exportOffground = false;

        while (!cmd.arguments().empty()) {
            const QString& ARGUMENT = cmd.arguments().front();
            if (ccCommandLineInterface::IsCommand(ARGUMENT,
                                                  COMMAND_CSF_SCENE)) {
                cmd.arguments().pop_front();
                bool conv = false;
                QString scene = cmd.arguments().takeFirst();
                if (scene == COMMAND_CSF_SCENE_SLOPE) {
                    cmd.print("Set scene to steep slope");
                    csfRigidness = 1;
                } else if (scene == COMMAND_CSF_SCENE_RELIEF) {
                    cmd.print("Set scene to relief");
                    csfRigidness = 2;
                } else if (scene == COMMAND_CSF_SCENE_FLAT) {
                    cmd.print("Set scene to flat");
                    csfRigidness = 3;
                } else {
                    cmd.error("Unknown scene parameter. Defaulting to relief");
                    csfRigidness = 2;
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARGUMENT, COMMAND_CSF_PROC_SLOPE)) {
                cmd.arguments().pop_front();
                cmd.print("Slope processing turned on");
                csfPostprocessing = true;
            } else if (ccCommandLineInterface::IsCommand(
                               ARGUMENT, COMMAND_CSF_CLOTH_RESOLUTION)) {
                cmd.arguments().pop_front();
                bool conv = false;
                clothResolution = cmd.arguments().takeFirst().toDouble(&conv);
                if (!conv) {
                    return cmd.error(
                            QObject::tr(
                                    "Invalid parameter: value after \"-%1\"")
                                    .arg(COMMAND_CSF_CLOTH_RESOLUTION));
                }
                cmd.print(QString("Custom cloth resulution set: %1")
                                  .arg(clothResolution));
            } else if (ccCommandLineInterface::IsCommand(
                               ARGUMENT, COMMAND_CSF_MAX_ITERATION)) {
                cmd.arguments().pop_front();
                bool conv = false;
                maxIteration = cmd.arguments().takeFirst().toInt(&conv);
                if (!conv) {
                    return cmd.error(
                            QObject::tr(
                                    "Invalid parameter: value after \"-%1\"")
                                    .arg(COMMAND_CSF_MAX_ITERATION));
                }
                cmd.print(QString("Custom max iteration set: %")
                                  .arg(maxIteration));
            } else if (ccCommandLineInterface::IsCommand(
                               ARGUMENT, COMMAND_CSF_CLASS_THRESHOLD)) {
                cmd.arguments().pop_front();
                bool conv = false;
                classThreshold = cmd.arguments().takeFirst().toDouble(&conv);
                if (!conv) {
                    return cmd.error(
                            QObject::tr(
                                    "Invalid parameter: value after \"-%1\"")
                                    .arg(COMMAND_CSF_CLASS_THRESHOLD));
                }
                cmd.print(QString("Custom class threshold set: %1")
                                  .arg(classThreshold));
            } else if (ccCommandLineInterface::IsCommand(
                               ARGUMENT, COMMAND_CSF_EXPORT_GROUND)) {
                cmd.arguments().pop_front();
                cmd.print("Ground will be exported");
                exportGround = true;
            } else if (ccCommandLineInterface::IsCommand(
                               ARGUMENT, COMMAND_CSF_EXPORT_OFFGROUND)) {
                cmd.arguments().pop_front();
                cmd.print("Off-ground will be exported");
                exportOffground = true;
            } else {
                cmd.print("Set all parameters");
                break;
            }
        }

        // instantiation a CSF class
        CSF csf(csfPC);

        // setup paramter
        csf.params.k_nearest_points = 1;
        csf.params.bSloopSmooth = csfPostprocessing;
        csf.params.time_step = 0.65;
        csf.params.class_threshold = classThreshold;
        csf.params.cloth_resolution = clothResolution;
        csf.params.rigidness = csfRigidness;
        csf.params.iterations = maxIteration;

        std::vector<int> groundIndexes;
        std::vector<int> offGroundIndexes;
        ccMesh* clothMesh = nullptr;
        if (!csf.do_filtering(groundIndexes, offGroundIndexes, false, clothMesh,
                              nullptr, cmd.widgetParent())) {
            return cmd.error("Process failed");
        }

        cmd.print(QString("[CSF] %1% of points classified as ground points")
                          .arg((groundIndexes.size() * 100.0) / count, 0, 'f',
                               2));

        // extract ground subset
        ccPointCloud* groundpoint = nullptr;
        {
            cloudViewer::ReferenceCloud groundpc(pc);
            if (groundpc.reserve(static_cast<unsigned>(groundIndexes.size()))) {
                for (unsigned j = 0; j < groundIndexes.size(); ++j) {
                    groundpc.addPointIndex(groundIndexes[j]);
                }
                groundpoint = pc->partialClone(&groundpc);
            }
        }
        if (!groundpoint) {
            cmd.print(
                    "Failed to extract the ground subset (not enough memory)");
        }

        ccPointCloud* offgroundpoint = nullptr;
        {
            cloudViewer::ReferenceCloud offgroundpc(pc);
            if (offgroundpc.reserve(
                        static_cast<unsigned>(offGroundIndexes.size()))) {
                for (unsigned k = 0; k < offGroundIndexes.size(); ++k) {
                    offgroundpc.addPointIndex(offGroundIndexes[k]);
                }
                offgroundpoint = pc->partialClone(&offgroundpc);
            }
        }
        if (!offgroundpoint) {
            cmd.print(
                    "Failed to extract the off-ground subset (not enough "
                    "memory)");
            if (!groundpoint) {
                return false;
            }
        }

        QString baseName = cmd.clouds()[0].basename;

        cmd.removeClouds();

        if (groundpoint) {
            CLCloudDesc groundDesc(groundpoint,
                                   baseName + QObject::tr("_ground_points"),
                                   -1);  // add cloud to the current pool
            cmd.clouds().push_back(groundDesc);
            if (exportGround) {
                QString errorStr = cmd.exportEntity(
                        groundDesc, QString(), nullptr,
                        ccCommandLineInterface::ExportOption::ForceNoTimestamp);
                if (!errorStr.isEmpty()) {
                    cmd.error(errorStr);
                }
            }
        }

        if (offgroundpoint) {
            CLCloudDesc offgroundDesc(
                    offgroundpoint, baseName + QObject::tr("_offground_points"),
                    -1);
            cmd.clouds().push_back(offgroundDesc);
            if (exportOffground) {
                QString errorStr = cmd.exportEntity(
                        offgroundDesc, QString(), nullptr,
                        ccCommandLineInterface::ExportOption::ForceNoTimestamp);
                if (!errorStr.isEmpty()) {
                    cmd.error(errorStr);
                }
            }
        }

        return true;
    }
};

#endif  // QCSF_PLUGIN_COMMANDS_HEADER
