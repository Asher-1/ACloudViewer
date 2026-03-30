// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvFacet.h>
#include <ecvHObject.h>

#include <QObject>

#include "ecvCommandLineInterface.h"
#include "facetsClassifier.h"
#include "qFacets.h"

static constexpr char COMMAND_FACETS[] = "FACETS";
static constexpr char COMMAND_FACETS_EXTRACT[] = "EXTRACT_FACETS";
static constexpr char COMMAND_FACETS_ALGO[] = "ALGO";
static constexpr char COMMAND_FACETS_KD_MAX_ANGLE[] =
        "KD_TREE_FUSION_MAX_ANGLE_DEG";
static constexpr char COMMAND_FACETS_KD_MAX_REL_DIST[] =
        "KD_TREE_FUSION_MAX_RELATIVE_DISTANCE";
static constexpr char COMMAND_FACETS_OCTREE_LEVEL[] = "OCTREE_LEVEL";
static constexpr char COMMAND_FACETS_RETRO_PROJ[] =
        "USE_RETRO_PROJECTION_ERROR";
static constexpr char COMMAND_FACETS_ERR_MAX[] = "ERROR_MAX_PER_FACET";
static constexpr char COMMAND_FACETS_MIN_PTS[] = "MIN_POINTS_PER_FACET";
static constexpr char COMMAND_FACETS_MAX_EDGE[] = "MAX_EDGE_LENGTH";
static constexpr char COMMAND_FACETS_ERR_MEASURE[] = "ERROR_MEASURE";
static constexpr char COMMAND_FACETS_CLASSIFY_ANGLE[] =
        "CLASSIFY_FACETS_BY_ANGLE";
static constexpr char COMMAND_FACETS_CLASSIF_STEP[] = "CLASSIF_ANGLE_STEP";
static constexpr char COMMAND_FACETS_CLASSIF_MAX_DIST[] = "CLASSIF_MAX_DIST";
static constexpr char COMMAND_FACETS_EXPORT_SHP[] = "EXPORT_FACETS";
static constexpr char COMMAND_FACETS_SHAPE_NAME[] = "SHAPE_FILENAME";
static constexpr char COMMAND_FACETS_USE_NATIVE[] = "USE_NATIVE_ORIENTATION";
static constexpr char COMMAND_FACETS_USE_GLOBAL[] = "USE_GLOBAL_ORIENTATION";
static constexpr char COMMAND_FACETS_USE_CUSTOM[] = "USE_CUSTOM_ORIENTATION";
static constexpr char COMMAND_FACETS_EXPORT_CSV[] = "EXPORT_FACETS_INFO";
static constexpr char COMMAND_FACETS_CSV_NAME[] = "CSV_FILENAME";
static constexpr char COMMAND_FACETS_COORDS_CSV[] = "COORDS_IN_CSV";

inline void commandFacetsCollectFacets(ccHObject* root,
                                       qFacets::FacetSet& out) {
    if (!root) {
        return;
    }
    if (root->isA(CV_TYPES::FACET)) {
        ccFacet* f = static_cast<ccFacet*>(root);
        if (f->getContour()) {
            out.insert(f);
        }
        return;
    }
    ccHObject::Container childFacets;
    root->filterChildren(childFacets, true, CV_TYPES::FACET);
    for (ccHObject* child : childFacets) {
        ccFacet* f = static_cast<ccFacet*>(child);
        if (f->getContour()) {
            out.insert(f);
        }
    }
}

struct CommandFacets : public ccCommandLineInterface::Command {
    CommandFacets()
        : ccCommandLineInterface::Command(QStringLiteral("FACETS"),
                                          QStringLiteral("FACETS")) {}

    bool process(ccCommandLineInterface& cmd) override {
        cmd.print("[FACETS]");

        if (cmd.clouds().empty()) {
            return cmd.error(QObject::tr("No point cloud loaded. Open one with "
                                         "\"-O\" before "
                                         "\"-%1\"")
                                     .arg(COMMAND_FACETS));
        }

        qFacets::FacetsParams params;

        while (!cmd.arguments().empty()) {
            const QString& ARG = cmd.arguments().front();
            if (ccCommandLineInterface::IsCommand(ARG,
                                                  COMMAND_FACETS_EXTRACT)) {
                cmd.arguments().pop_front();
                params.extractFacets = true;
            } else if (ccCommandLineInterface::IsCommand(ARG,
                                                         COMMAND_FACETS_ALGO)) {
                cmd.arguments().pop_front();
                if (cmd.arguments().empty()) {
                    return cmd.error(
                            QObject::tr("Missing value after \"-%1\" (expected "
                                        "ALGO_KD_TREE or ALGO_FAST_MARCHING)")
                                    .arg(COMMAND_FACETS_ALGO));
                }
                QString v = cmd.arguments().takeFirst().toUpper();
                if (v == QStringLiteral("ALGO_KD_TREE")) {
                    params.algo = CellsFusionDlg::ALGO_KD_TREE;
                } else if (v == QStringLiteral("ALGO_FAST_MARCHING")) {
                    params.algo = CellsFusionDlg::ALGO_FAST_MARCHING;
                } else {
                    return cmd.error(QObject::tr("Invalid \"-%1\" value: %2")
                                             .arg(COMMAND_FACETS_ALGO, v));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_KD_MAX_ANGLE)) {
                cmd.arguments().pop_front();
                bool ok = false;
                params.kdTreeFusionMaxAngleDeg =
                        cmd.arguments().takeFirst().toDouble(&ok);
                if (!ok) {
                    return cmd.error(QObject::tr("Invalid number after \"-%1\"")
                                             .arg(COMMAND_FACETS_KD_MAX_ANGLE));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_KD_MAX_REL_DIST)) {
                cmd.arguments().pop_front();
                bool ok = false;
                params.kdTreeFusionMaxRelativeDistance =
                        cmd.arguments().takeFirst().toDouble(&ok);
                if (!ok) {
                    return cmd.error(
                            QObject::tr("Invalid number after \"-%1\"")
                                    .arg(COMMAND_FACETS_KD_MAX_REL_DIST));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_OCTREE_LEVEL)) {
                cmd.arguments().pop_front();
                bool ok = false;
                params.octreeLevel = cmd.arguments().takeFirst().toUInt(&ok);
                if (!ok) {
                    return cmd.error(
                            QObject::tr(
                                    "Invalid unsigned integer after \"-%1\"")
                                    .arg(COMMAND_FACETS_OCTREE_LEVEL));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_RETRO_PROJ)) {
                cmd.arguments().pop_front();
                params.useRetroProjectionError = true;
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_ERR_MAX)) {
                cmd.arguments().pop_front();
                bool ok = false;
                params.errorMaxPerFacet =
                        cmd.arguments().takeFirst().toDouble(&ok);
                if (!ok) {
                    return cmd.error(QObject::tr("Invalid number after \"-%1\"")
                                             .arg(COMMAND_FACETS_ERR_MAX));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_MIN_PTS)) {
                cmd.arguments().pop_front();
                bool ok = false;
                params.minPointsPerFacet =
                        cmd.arguments().takeFirst().toUInt(&ok);
                if (!ok) {
                    return cmd.error(
                            QObject::tr(
                                    "Invalid unsigned integer after \"-%1\"")
                                    .arg(COMMAND_FACETS_MIN_PTS));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_MAX_EDGE)) {
                cmd.arguments().pop_front();
                bool ok = false;
                params.maxEdgeLength =
                        cmd.arguments().takeFirst().toDouble(&ok);
                if (!ok) {
                    return cmd.error(QObject::tr("Invalid number after \"-%1\"")
                                             .arg(COMMAND_FACETS_MAX_EDGE));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_ERR_MEASURE)) {
                cmd.arguments().pop_front();
                if (cmd.arguments().empty()) {
                    return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                             .arg(COMMAND_FACETS_ERR_MEASURE));
                }
                QString m = cmd.arguments().takeFirst().toUpper();
                if (m == QStringLiteral("RMS")) {
                    params.errorMeasure =
                            cloudViewer::DistanceComputationTools::RMS;
                } else if (m == QStringLiteral("MAX_DIST_68_PERCENT")) {
                    params.errorMeasure = cloudViewer::
                            DistanceComputationTools::MAX_DIST_68_PERCENT;
                } else if (m == QStringLiteral("MAX_DIST_95_PERCENT")) {
                    params.errorMeasure = cloudViewer::
                            DistanceComputationTools::MAX_DIST_95_PERCENT;
                } else if (m == QStringLiteral("MAX_DIST_99_PERCENT")) {
                    params.errorMeasure = cloudViewer::
                            DistanceComputationTools::MAX_DIST_99_PERCENT;
                } else if (m == QStringLiteral("MAX_DIST")) {
                    params.errorMeasure =
                            cloudViewer::DistanceComputationTools::MAX_DIST;
                } else {
                    return cmd.error(
                            QObject::tr("Invalid \"-%1\" value: %2")
                                    .arg(COMMAND_FACETS_ERR_MEASURE, m));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_CLASSIFY_ANGLE)) {
                cmd.arguments().pop_front();
                params.classifyFacetsByAngle = true;
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_CLASSIF_STEP)) {
                cmd.arguments().pop_front();
                bool ok = false;
                params.classifAngleStep =
                        cmd.arguments().takeFirst().toDouble(&ok);
                if (!ok) {
                    return cmd.error(QObject::tr("Invalid number after \"-%1\"")
                                             .arg(COMMAND_FACETS_CLASSIF_STEP));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_CLASSIF_MAX_DIST)) {
                cmd.arguments().pop_front();
                bool ok = false;
                params.classifMaxDist =
                        cmd.arguments().takeFirst().toDouble(&ok);
                if (!ok) {
                    return cmd.error(
                            QObject::tr("Invalid number after \"-%1\"")
                                    .arg(COMMAND_FACETS_CLASSIF_MAX_DIST));
                }
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_EXPORT_SHP)) {
                cmd.arguments().pop_front();
                params.exportFacets = true;
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_SHAPE_NAME)) {
                cmd.arguments().pop_front();
                if (cmd.arguments().empty()) {
                    return cmd.error(QObject::tr("Missing path after \"-%1\"")
                                             .arg(COMMAND_FACETS_SHAPE_NAME));
                }
                params.shapeFilename = cmd.arguments().takeFirst();
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_USE_NATIVE)) {
                cmd.arguments().pop_front();
                params.useNativeOrientation = true;
                params.useGlobalOrientation = false;
                params.useCustomOrientation = false;
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_USE_GLOBAL)) {
                cmd.arguments().pop_front();
                params.useNativeOrientation = false;
                params.useGlobalOrientation = true;
                params.useCustomOrientation = false;
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_USE_CUSTOM)) {
                cmd.arguments().pop_front();
                if (cmd.arguments().size() < 3) {
                    return cmd.error(
                            QObject::tr("Expected Nx Ny Nz after \"-%1\"")
                                    .arg(COMMAND_FACETS_USE_CUSTOM));
                }
                bool ok = true;
                params.nX = cmd.arguments().takeFirst().toDouble(&ok);
                params.nY = cmd.arguments().takeFirst().toDouble(&ok);
                params.nZ = cmd.arguments().takeFirst().toDouble(&ok);
                if (!ok) {
                    return cmd.error(
                            QObject::tr("Invalid numbers after \"-%1\"")
                                    .arg(COMMAND_FACETS_USE_CUSTOM));
                }
                params.useNativeOrientation = false;
                params.useGlobalOrientation = false;
                params.useCustomOrientation = true;
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_EXPORT_CSV)) {
                cmd.arguments().pop_front();
                params.exportFacetsInfo = true;
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_CSV_NAME)) {
                cmd.arguments().pop_front();
                if (cmd.arguments().empty()) {
                    return cmd.error(QObject::tr("Missing path after \"-%1\"")
                                             .arg(COMMAND_FACETS_CSV_NAME));
                }
                params.csvFilename = cmd.arguments().takeFirst();
            } else if (ccCommandLineInterface::IsCommand(
                               ARG, COMMAND_FACETS_COORDS_CSV)) {
                cmd.arguments().pop_front();
                params.coordsInCsv = true;
            } else {
                break;
            }
        }

        if (!params.extractFacets) {
            return cmd.error(QObject::tr("Facet extraction disabled: add "
                                         "\"-%1\" to run extraction.")
                                     .arg(COMMAND_FACETS_EXTRACT));
        }

        CLCloudDesc& cld = cmd.clouds().front();
        ccPointCloud* pc = cld.pc;
        if (!pc) {
            return cmd.error(QObject::tr("Invalid cloud"));
        }

        ecvProgressDialog* progressCb = cmd.progressDialog();
        bool facetError = false;
        ccHObject* group = qFacets::ExecuteFacetExtraction(
                pc, params, facetError, progressCb);

        if (!group) {
            return cmd.error(QObject::tr(
                    "Facet extraction failed or produced no facets."));
        }

        if (facetError) {
            cmd.warning(
                    QObject::tr("Warning: errors occurred during facet "
                                "creation; result may "
                                "be incomplete."));
        }

        if (params.classifyFacetsByAngle) {
            if (!FacetsClassifier::ByOrientation(group, params.classifAngleStep,
                                                 params.classifMaxDist)) {
                cmd.warning(QObject::tr(
                        "Facet classification by orientation failed."));
            }
        }

        qFacets::FacetSet facetSet;
        commandFacetsCollectFacets(group, facetSet);

        const bool silent = cmd.silentMode();

        if (params.exportFacets) {
            if (!qFacets::ExecuteExportFacets(facetSet, params.shapeFilename,
                                              params.useNativeOrientation,
                                              params.useGlobalOrientation,
                                              params.useCustomOrientation,
                                              params.nX, params.nY, params.nZ,
                                              silent)) {
                delete group;
                return cmd.error(QObject::tr("Shapefile export failed."));
            }
        }

        if (params.exportFacetsInfo) {
            if (!qFacets::ExecuteExportFacetsInfo(
                        facetSet, params.csvFilename, params.coordsInCsv,
                        params.useNativeOrientation,
                        params.useGlobalOrientation,
                        params.useCustomOrientation, params.nX, params.nY,
                        params.nZ, silent)) {
                delete group;
                return cmd.error(QObject::tr("CSV export failed."));
            }
        }

        if (cmd.autoSaveMode()) {
            CLGroupDesc grpDesc(group, cld.basename + QObject::tr("_FACETS"),
                                cld.path);
            QString errStr = cmd.exportEntity(
                    grpDesc, QString(), nullptr,
                    ccCommandLineInterface::ExportOption::ForceHierarchy);
            if (!errStr.isEmpty()) {
                cmd.warning(errStr);
            }
        }

        delete group;
        return true;
    }
};
