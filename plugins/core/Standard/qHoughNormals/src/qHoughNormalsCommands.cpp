// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qHoughNormalsCommands.h"

#include "Normals.h"

#include <ecvPointCloud.h>

#include <QObject>

static const char COMMAND_HOUGH_NORMALS[] = "HOUGH_NORMALS";
static const char COMMAND_HN_K[] = "K";
static const char COMMAND_HN_T[] = "T";
static const char COMMAND_HN_N_PHI[] = "N_PHI";
static const char COMMAND_HN_N_ROT[] = "N_ROT";
static const char COMMAND_HN_TOL_ANGLE[] = "TOL_ANGLE_RAD";
static const char COMMAND_HN_K_DENSITY[] = "K_DENSITY";
static const char COMMAND_HN_USE_DENSITY[] = "USE_DENSITY";

CommandHoughNormals::CommandHoughNormals()
        : ccCommandLineInterface::Command("Hough Normals", COMMAND_HOUGH_NORMALS) {
}

bool CommandHoughNormals::process(ccCommandLineInterface& cmd) {
    cmd.print("[HOUGH_NORMALS]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_HOUGH_NORMALS));
    }

    int K = 100;
    int T = 1000;
    int n_phi = 15;
    int n_rot = 5;
    double tol_angle_rad = 0.79;
    int k_density = 5;
    bool use_density = false;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_HN_K)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_HN_K));
            bool ok;
            K = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok) return cmd.error("Invalid value for -K");
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_HN_T)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_HN_T));
            bool ok;
            T = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok) return cmd.error("Invalid value for -T");
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_HN_N_PHI)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_HN_N_PHI));
            bool ok;
            n_phi = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok) return cmd.error("Invalid value for -N_PHI");
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_HN_N_ROT)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_HN_N_ROT));
            bool ok;
            n_rot = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok) return cmd.error("Invalid value for -N_ROT");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                       COMMAND_HN_TOL_ANGLE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_HN_TOL_ANGLE));
            bool ok;
            tol_angle_rad = cmd.arguments().takeFirst().toDouble(&ok);
            if (!ok) return cmd.error("Invalid value for -TOL_ANGLE_RAD");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                     COMMAND_HN_K_DENSITY)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_HN_K_DENSITY));
            bool ok;
            k_density = cmd.arguments().takeFirst().toInt(&ok);
            if (!ok) return cmd.error("Invalid value for -K_DENSITY");
        } else if (ccCommandLineInterface::IsCommand(arg,
                                                       COMMAND_HN_USE_DENSITY)) {
            cmd.arguments().pop_front();
            use_density = true;
        } else {
            break;
        }
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* cloud = desc.pc;
        if (!cloud) continue;

        cmd.print(QObject::tr("[HOUGH_NORMALS] Processing cloud '%1' (%2 points)")
                          .arg(cloud->getName())
                          .arg(cloud->size()));

        size_t pointCount = cloud->size();
        Eigen::MatrixX3d pc;
        pc.resize(pointCount, 3);
        for (size_t i = 0; i < pointCount; ++i) {
            const CCVector3* P = cloud->getPoint(static_cast<unsigned>(i));
            pc.row(i) = Eigen::Vector3d(P->x, P->y, P->z);
        }

        Eigen::MatrixX3d normals;
        Eigen_Normal_Estimator ne(pc, normals);
        ne.get_K() = static_cast<size_t>(K);
        ne.get_T() = T;
        ne.density_sensitive() = use_density;
        ne.get_n_phi() = n_phi;
        ne.get_n_rot() = n_rot;
        ne.get_tol_angle_rad() = tol_angle_rad;
        ne.get_K_density() = static_cast<size_t>(k_density);

        ne.estimate_normals();

        if (!cloud->resizeTheNormsTable()) {
            return cmd.error("Not enough memory to store normals");
        }

        for (size_t i = 0; i < pointCount; ++i) {
            const Eigen::Vector3d& n = normals.row(i);
            CCVector3 N(static_cast<PointCoordinateType>(n.x()),
                        static_cast<PointCoordinateType>(n.y()),
                        static_cast<PointCoordinateType>(n.z()));
            cloud->setPointNormal(static_cast<unsigned>(i), N);
        }

        cloud->showNormals(true);
        desc.basename += QString("_HOUGH_NORMALS");

        if (cmd.autoSaveMode()) {
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }
    return true;
}
