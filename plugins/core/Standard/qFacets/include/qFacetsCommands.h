// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDir>
#include <QFileInfo>
#include <QObject>

#include "ecvCommandLineInterface.h"
#include "facetsClassifier.h"
#include "qFacets.h"

static constexpr char COMMAND_FACETS[] = "FACETS";

static constexpr char EXTRACT_FACETS[] = "-EXTRACT_FACETS";
static constexpr char ALGO[] = "-ALGO";
static constexpr char ALGO_FAST_MARCHING[] = "ALGO_FAST_MARCHING";
static constexpr char ALGO_KD_TREE[] = "ALGO_KD_TREE";
static constexpr char KD_TREE_FUSION_MAX_ANGLE_DEG[] =
        "-KD_TREE_FUSION_MAX_ANGLE_DEG";
static constexpr char KD_TREE_FUSION_MAX_RELATIVE_DISTANCE[] =
        "-KD_TREE_FUSION_MAX_RELATIVE_DISTANCE";
static constexpr char OCTREE_LEVEL[] = "-OCTREE_LEVEL";
static constexpr char USE_RETRO_PROJECTION_ERROR[] =
        "-USE_RETRO_PROJECTION_ERROR";
static constexpr char ERROR_MAX_PER_FACET[] = "-ERROR_MAX_PER_FACET";
static constexpr char MAX_EDGE_LENGTH[] = "-MAX_EDGE_LENGTH";
static constexpr char MIN_POINTS_PER_FACET[] = "MIN_POINTS_PER_FACET";
static constexpr char ERROR_MEASURE[] = "-ERROR_MEASURE";
static constexpr char RMS[] = "RMS";
static constexpr char MAX_DIST_68_PERCENT[] = "MAX_DIST_68_PERCENT";
static constexpr char MAX_DIST_95_PERCENT[] = "MAX_DIST_95_PERCENT";
static constexpr char MAX_DIST_99_PERCENT[] = "MAX_DIST_99_PERCENT";
static constexpr char MAX_DIST[] = "MAX_DIST";
static constexpr char CLASSIFY_FACETS_BY_ANGLE[] = "-CLASSIFY_FACETS_BY_ANGLE";
static constexpr char CLASSIF_ANGLE_STEP[] = "-CLASSIF_ANGLE_STEP";
static constexpr char CLASSIF_MAX_DIST[] = "-CLASSIF_MAX_DIST";
static constexpr char EXPORT_FACETS[] = "-EXPORT_FACETS";
static constexpr char SHAPE_FILENAME[] = "-SHAPE_FILENAME";
static constexpr char USE_NATIVE_ORIENTATION[] = "-USE_NATIVE_ORIENTATION";
static constexpr char USE_GLOBAL_ORIENTATION[] = "-USE_GLOBAL_ORIENTATION";
static constexpr char USE_CUSTOM_ORIENTATION[] = "-USE_CUSTOM_ORIENTATION";
static constexpr char EXPORT_FACETS_INFO[] = "-EXPORT_FACETS_INFO";
static constexpr char CSV_FILENAME[] = "-CSV_FILENAME";
static constexpr char COORDS_IN_CSV[] = "-COORDS_IN_CSV";

static void AddFacetsToGroup(ccHObject* group,
                             const qFacets::FacetSet& facets) {
    for (ccFacet* facet : facets) {
        if (facet) {
            group->addChild(facet);
        }
    }
}

static void GetFacetsFromGroup(ccHObject* group, qFacets::FacetSet& facets) {
    facets.clear();
    ccHObject::Container childFacets;
    group->filterChildren(childFacets, true, CV_TYPES::FACET);

    for (ccHObject* childFacet : childFacets) {
        ccFacet* facet = static_cast<ccFacet*>(childFacet);
        if (facet->getContour()) {
            facets.insert(facet);
        }
    }
}

struct CommandFacets : public ccCommandLineInterface::Command {
    CommandFacets()
        : ccCommandLineInterface::Command(QObject::tr("FACETS"),
                                          COMMAND_FACETS) {}

    bool process(ccCommandLineInterface& cmd) override {
        cmd.print("[FACETS]");
        if (cmd.clouds().empty()) {
            return cmd.error(
                    QObject::tr("No point cloud to attempt FACETS on (be sure "
                                "to open one with \"-O [cloud filename]\" "
                                "before \"-%2\")")
                            .arg(COMMAND_FACETS));
        }

        qFacets::FacetsParams params;
        QStringList paramNames =
                QStringList()
                << EXTRACT_FACETS << ALGO << KD_TREE_FUSION_MAX_ANGLE_DEG
                << KD_TREE_FUSION_MAX_RELATIVE_DISTANCE << OCTREE_LEVEL
                << USE_RETRO_PROJECTION_ERROR << ERROR_MAX_PER_FACET
                << MIN_POINTS_PER_FACET << MAX_EDGE_LENGTH << ERROR_MEASURE
                << CLASSIFY_FACETS_BY_ANGLE << CLASSIF_ANGLE_STEP
                << CLASSIF_MAX_DIST << EXPORT_FACETS << SHAPE_FILENAME
                << USE_NATIVE_ORIENTATION << USE_GLOBAL_ORIENTATION
                << USE_CUSTOM_ORIENTATION << EXPORT_FACETS_INFO << CSV_FILENAME
                << COORDS_IN_CSV;
        QStringList algoNames = QStringList()
                                << ALGO_FAST_MARCHING << ALGO_KD_TREE;
        QStringList errorMeasureNames = QStringList()
                                        << RMS << MAX_DIST_68_PERCENT
                                        << MAX_DIST_95_PERCENT
                                        << MAX_DIST_99_PERCENT << MAX_DIST;

        if (!cmd.arguments().empty()) {
            QString param = cmd.arguments().takeFirst().toUpper();
            while (paramNames.contains(param)) {
                cmd.print(QObject::tr("\t%1 : %2").arg(COMMAND_FACETS, param));

                if (param == EXTRACT_FACETS) {
                    params.extractFacets = true;
                } else if (param == ALGO) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: Algorithm type "
                                            "after \"-%1 %2\"")
                                        .arg(COMMAND_FACETS, ALGO));
                    }

                    QString val = cmd.arguments().takeFirst().toUpper();
                    cmd.print(QObject::tr("\t-ALGO : %1").arg(val));
                    if (val == ALGO_FAST_MARCHING) {
                        params.algo = CellsFusionDlg::ALGO_FAST_MARCHING;
                    } else if (val == ALGO_KD_TREE) {
                        params.algo = CellsFusionDlg::ALGO_KD_TREE;
                    } else {
                        return cmd.error(
                                QObject::tr("No valid parameter: Algorithm "
                                            "type after \"-%1 %2\"")
                                        .arg(COMMAND_FACETS, ALGO));
                    }
                } else if (param == KD_TREE_FUSION_MAX_ANGLE_DEG) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: number after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS,
                                             KD_TREE_FUSION_MAX_ANGLE_DEG));
                    }
                    bool ok = false;
                    float val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for "
                                "-KD_TREE_FUSION_MAX_ANGLE_DEG!");
                    }
                    cmd.print(
                            QObject::tr("\t-KD_TREE_FUSION_MAX_ANGLE_DEG : %1")
                                    .arg(val));
                    params.kdTreeFusionMaxAngleDeg = val;
                } else if (param == KD_TREE_FUSION_MAX_RELATIVE_DISTANCE) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: number after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS,
                                             KD_TREE_FUSION_MAX_RELATIVE_DISTANCE));
                    }
                    bool ok = false;
                    float val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for "
                                "-KD_TREE_FUSION_MAX_RELATIVE_DISTANCE!");
                    }
                    cmd.print(QObject::tr("\t-KD_TREE_FUSION_MAX_RELATIVE_"
                                          "DISTANCE : %1")
                                      .arg(val));
                    params.kdTreeFusionMaxRelativeDistance = val;
                } else if (param == OCTREE_LEVEL) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: number after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS, OCTREE_LEVEL));
                    }
                    bool ok = false;
                    int val = cmd.arguments().takeFirst().toInt(&ok);
                    if (!ok) {
                        return cmd.error("Invalid number for -OCTREE_LEVEL!");
                    }
                    cmd.print(QObject::tr("\t-OCTREE_LEVEL : %1").arg(val));
                    params.octreeLevel = static_cast<unsigned>(val);
                } else if (param == USE_RETRO_PROJECTION_ERROR) {
                    params.useRetroProjectionError = true;
                } else if (param == ERROR_MAX_PER_FACET) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: number after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS,
                                             ERROR_MAX_PER_FACET));
                    }
                    bool ok = false;
                    float val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for -ERROR_MAX_PER_FACET!");
                    }
                    cmd.print(QObject::tr("\t-ERROR_MAX_PER_FACET : %1")
                                      .arg(val));
                    params.errorMaxPerFacet = val;
                } else if (param == MIN_POINTS_PER_FACET) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: number after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS,
                                             MIN_POINTS_PER_FACET));
                    }
                    bool ok = false;
                    int val = cmd.arguments().takeFirst().toInt(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for -MIN_POINTS_PER_FACET!");
                    }
                    cmd.print(QObject::tr("\t-MIN_POINTS_PER_FACET : %1")
                                      .arg(val));
                    params.minPointsPerFacet = static_cast<unsigned>(val);
                } else if (param == MAX_EDGE_LENGTH) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: number after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS, MAX_EDGE_LENGTH));
                    }
                    bool ok = false;
                    float val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for -MAX_EDGE_LENGTH!");
                    }
                    cmd.print(QObject::tr("\t-MAX_EDGE_LENGTH : %1").arg(val));
                    params.maxEdgeLength = val;
                } else if (param == ERROR_MEASURE) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: ERROR_MEASURE "
                                            "type after \"-%1 %2\"")
                                        .arg(COMMAND_FACETS, ERROR_MEASURE));
                    }

                    QString val = cmd.arguments().takeFirst().toUpper();
                    cmd.print(QObject::tr("\t-ERROR_MEASURE : %1").arg(val));
                    if (val == RMS) {
                        params.errorMeasure =
                                cloudViewer::DistanceComputationTools::RMS;
                    } else if (val == MAX_DIST_68_PERCENT) {
                        params.errorMeasure = cloudViewer::
                                DistanceComputationTools::MAX_DIST_68_PERCENT;
                    } else if (val == MAX_DIST_95_PERCENT) {
                        params.errorMeasure = cloudViewer::
                                DistanceComputationTools::MAX_DIST_95_PERCENT;
                    } else if (val == MAX_DIST_99_PERCENT) {
                        params.errorMeasure = cloudViewer::
                                DistanceComputationTools::MAX_DIST_99_PERCENT;
                    } else if (val == MAX_DIST) {
                        params.errorMeasure =
                                cloudViewer::DistanceComputationTools::MAX_DIST;
                    } else {
                        return cmd.error(
                                QObject::tr("No valid parameter: Error measure "
                                            "type after \"-%1 %2\"")
                                        .arg(COMMAND_FACETS, ERROR_MEASURE));
                    }
                } else if (param == CLASSIFY_FACETS_BY_ANGLE) {
                    params.classifyFacetsByAngle = true;
                } else if (param == CLASSIF_ANGLE_STEP) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: number after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS,
                                             CLASSIF_ANGLE_STEP));
                    }
                    bool ok = false;
                    float val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for -CLASSIF_ANGLE_STEP!");
                    }
                    cmd.print(
                            QObject::tr("\t-CLASSIF_ANGLE_STEP : %1").arg(val));
                    params.classifAngleStep = val;
                } else if (param == CLASSIF_MAX_DIST) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: number after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS, CLASSIF_MAX_DIST));
                    }
                    bool ok = false;
                    float val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for -CLASSIF_MAX_DIST!");
                    }
                    cmd.print(QObject::tr("\t-CLASSIF_MAX_DIST : %1").arg(val));
                    params.classifMaxDist = val;
                } else if (param == EXPORT_FACETS) {
                    params.exportFacets = true;
                } else if (param == SHAPE_FILENAME) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: filepath after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS, SHAPE_FILENAME));
                    }

                    QString val = cmd.arguments().takeFirst();
                    cmd.print(QObject::tr("\t-SHAPE_FILENAME : %1").arg(val));
                    params.shapeFilename = val;
                } else if (param == USE_NATIVE_ORIENTATION) {
                    params.useNativeOrientation = true;
                    params.useGlobalOrientation = false;
                    params.useCustomOrientation = false;
                } else if (param == USE_GLOBAL_ORIENTATION) {
                    params.useNativeOrientation = false;
                    params.useGlobalOrientation = true;
                    params.useCustomOrientation = false;
                } else if (param == USE_CUSTOM_ORIENTATION) {
                    params.useNativeOrientation = false;
                    params.useGlobalOrientation = false;
                    params.useCustomOrientation = true;

                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: Nx \"-%1 %2\"")
                                        .arg(COMMAND_FACETS,
                                             USE_CUSTOM_ORIENTATION));
                    }

                    bool ok = false;
                    float val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for Nx in "
                                "-USE_CUSTOM_ORIENTATION!");
                    }
                    params.nX = val;

                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: Ny \"-%1 %2\"")
                                        .arg(COMMAND_FACETS,
                                             USE_CUSTOM_ORIENTATION));
                    }

                    val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for Ny in "
                                "-USE_CUSTOM_ORIENTATION!");
                    }
                    params.nY = val;

                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: Nz \"-%1 %2\"")
                                        .arg(COMMAND_FACETS,
                                             USE_CUSTOM_ORIENTATION));
                    }
                    val = cmd.arguments().takeFirst().toFloat(&ok);
                    if (!ok) {
                        return cmd.error(
                                "Invalid number for Nz in "
                                "-USE_CUSTOM_ORIENTATION!");
                    }
                    params.nZ = val;
                } else if (param == EXPORT_FACETS_INFO) {
                    params.exportFacetsInfo = true;
                } else if (param == CSV_FILENAME) {
                    if (cmd.arguments().empty()) {
                        return cmd.error(
                                QObject::tr("Missing parameter: filepath after "
                                            "\"-%1 %2\"")
                                        .arg(COMMAND_FACETS, CSV_FILENAME));
                    }

                    QString val = cmd.arguments().takeFirst();
                    cmd.print(QObject::tr("\t-CSV_FILENAME : %1").arg(val));
                    params.csvFilename = val;
                } else if (param == COORDS_IN_CSV) {
                    params.coordsInCsv = true;
                }

                if (cmd.arguments().empty()) {
                    break;
                }
                param = cmd.arguments().takeFirst().toUpper();
            }

            if (!paramNames.contains(param)) {
                cmd.arguments().push_front(param);
            }
        }

        if (!params.extractFacets && !params.classifyFacetsByAngle &&
            !params.exportFacets && !params.exportFacetsInfo) {
            return cmd.error(
                    QObject::tr(
                            "No valid parameter: Need one of -%2, -%3, -%4, "
                            "or -%5 after \"-%1\"")
                            .arg(COMMAND_FACETS, EXTRACT_FACETS,
                                 CLASSIFY_FACETS_BY_ANGLE, EXPORT_FACETS,
                                 EXPORT_FACETS_INFO));
        }

        cmd.print(QObject::tr("[FACETS] Parameters including default"));
        if (params.extractFacets) {
            cmd.print(QObject::tr("\t-EXTRACT_FACETS"));
            if (params.algo == CellsFusionDlg::ALGO_KD_TREE) {
                cmd.print(QObject::tr("\t\t-ALGO : ALGO_KD_TREE"));
                cmd.print(
                        QObject::tr("\t\t\t-KD_TREE_FUSION_MAX_ANGLE_DEG : %1")
                                .arg(params.kdTreeFusionMaxAngleDeg));
                cmd.print(QObject::tr("\t\t\t-KD_TREE_FUSION_MAX_RELATIVE_"
                                      "DISTANCE : %1")
                                  .arg(params.kdTreeFusionMaxRelativeDistance));
            } else if (params.algo == CellsFusionDlg::ALGO_FAST_MARCHING) {
                cmd.print(QObject::tr("\t\t-ALGO ALGO_FAST_MARCHING"));
                cmd.print(QObject::tr("\t\t\t-OCTREE_LEVEL : %1")
                                  .arg(params.octreeLevel));
                cmd.print(QObject::tr("\t\t\t-USE_RETRO_PROJECTION_ERROR : %1")
                                  .arg(params.useRetroProjectionError));
            }

            if (params.errorMeasure ==
                cloudViewer::DistanceComputationTools::RMS) {
                cmd.print(QObject::tr("\t\t-ERROR_MEASURE RMS"));
            } else if (params.errorMeasure ==
                       cloudViewer::DistanceComputationTools::
                               MAX_DIST_68_PERCENT) {
                cmd.print(
                        QObject::tr("\t\t-ERROR_MEASURE MAX_DIST_68_PERCENT"));
            } else if (params.errorMeasure ==
                       cloudViewer::DistanceComputationTools::
                               MAX_DIST_95_PERCENT) {
                cmd.print(
                        QObject::tr("\t\t-ERROR_MEASURE MAX_DIST_95_PERCENT"));
            } else if (params.errorMeasure ==
                       cloudViewer::DistanceComputationTools::
                               MAX_DIST_99_PERCENT) {
                cmd.print(
                        QObject::tr("\t\t-ERROR_MEASURE MAX_DIST_99_PERCENT"));
            } else if (params.errorMeasure ==
                       cloudViewer::DistanceComputationTools::MAX_DIST) {
                cmd.print(QObject::tr("\t\t-ERROR_MEASURE MAX_DIST"));
            }
            cmd.print(QObject::tr("\t\t-ERROR_MAX_PER_FACET : %1")
                              .arg(params.errorMaxPerFacet));
            cmd.print(QObject::tr("\t\t-MIN_POINTS_PER_FACET : %1")
                              .arg(params.minPointsPerFacet));
            cmd.print(QObject::tr("\t\t-MAX_EDGE_LENGTH : %1")
                              .arg(params.maxEdgeLength));
        }

        if (params.classifyFacetsByAngle) {
            cmd.print(QObject::tr("\t-CLASSIFY_FACETS_BY_ANGLE"));
            cmd.print(QObject::tr("\t\t-CLASSIF_ANGLE_STEP : %1")
                              .arg(params.classifAngleStep));
            cmd.print(QObject::tr("\t\t-CLASSIF_MAX_DIST : %1")
                              .arg(params.classifMaxDist));
        }

        if (params.exportFacets) {
            cmd.print(QObject::tr("\t-EXPORT_FACETS"));
            cmd.print(QObject::tr("\t\t-SHAPE_FILENAME : \"%1\"")
                              .arg(params.shapeFilename));
            if (params.useNativeOrientation) {
                cmd.print(QObject::tr("\t\t-USE_NATIVE_ORIENTATION"));
            } else if (params.useGlobalOrientation) {
                cmd.print(QObject::tr("\t\t-USE_GLOBAL_ORIENTATION"));
            } else if (params.useCustomOrientation) {
                cmd.print(QObject::tr("\t\t-USE_CUSTOM_ORIENTATION : %1 %2 %3")
                                  .arg(params.nX)
                                  .arg(params.nY)
                                  .arg(params.nZ));
            }
        }

        if (params.exportFacetsInfo) {
            cmd.print(QObject::tr("\t-EXPORT_FACETS_INFO"));
            cmd.print(QObject::tr("\t\t-CSV_FILENAME : \"%1\"")
                              .arg(params.csvFilename));
            if (params.coordsInCsv) {
                cmd.print(QObject::tr("\t\t-COORDS_IN_CSV"));
                if (params.useNativeOrientation) {
                    cmd.print(QObject::tr("\t\t\t-USE_NATIVE_ORIENTATION"));
                } else if (params.useGlobalOrientation) {
                    cmd.print(QObject::tr("\t\t\t-USE_GLOBAL_ORIENTATION"));
                } else if (params.useCustomOrientation) {
                    cmd.print(
                            QObject::tr(
                                    "\t\t\t-USE_CUSTOM_ORIENTATION : %1 %2 %3")
                                    .arg(params.nX)
                                    .arg(params.nY)
                                    .arg(params.nZ));
                }
            }
        }

        ecvProgressDialog* progressCb = cmd.progressDialog();

        for (CLCloudDesc clCloud : cmd.clouds()) {
            qFacets::FacetSet facets;
            ccHObject* group = nullptr;

            if (params.extractFacets) {
                cmd.print(QObject::tr("[FACETS] Extracting Facets: \"%1\"")
                                  .arg(clCloud.pc->getName()));
                bool errorDuringFacetCreation = false;
                group = qFacets::ExecuteFacetExtraction(
                        clCloud.pc, params, errorDuringFacetCreation,
                        progressCb);

                if (errorDuringFacetCreation) {
                    cmd.error(
                            QObject::tr("[FACETS] Failed to extract facets."));
                    return false;
                }

                GetFacetsFromGroup(group, facets);
                if (facets.empty()) {
                    delete group;
                    cmd.error(QObject::tr(
                            "[FACETS] Did not extract any facets."));
                    return false;
                }
                cmd.print(QObject::tr("[FACETS] Extracted %1 facets")
                                  .arg(facets.size()));
            }

            if (params.classifyFacetsByAngle) {
                cmd.print(
                        QObject::tr("[FACETS] Classifying facets by angles."));
                if (facets.empty()) {
                    delete group;
                    cmd.error(
                            QObject::tr("[FACETS] Need facets. Must use "
                                        "-EXTRACT_FACETS."));
                    return false;
                }

                ccHObject* classifGroup =
                        new ccHObject(QString("FACETS group"));
                AddFacetsToGroup(classifGroup, facets);

                bool success = FacetsClassifier::ByOrientation(
                        classifGroup, params.classifAngleStep,
                        params.classifMaxDist);
                if (!success) {
                    delete classifGroup;
                    delete group;
                    cmd.error(QObject::tr(
                            "[FACETS] Failed to Classify facets by angles."));
                    return false;
                }
                delete classifGroup;
            }

            if (params.exportFacets) {
                cmd.print(QObject::tr(
                        "[FACETS] Exporting Facets info to shape file"));
                if (facets.empty()) {
                    delete group;
                    cmd.error(QObject::tr(
                            "[FACETS] Need facets. Must use -EXTRACT_FACETS."));
                    return false;
                }

                QFileInfo fileInfo(params.shapeFilename);
                QString directoryPath = fileInfo.path();
                QString newFileName = clCloud.pc->getName() + QString("_") +
                                      fileInfo.fileName();
                QString outputName = QDir(directoryPath).filePath(newFileName);
                QDir dir;
                if (!dir.mkpath(directoryPath)) {
                    delete group;
                    cmd.error(
                            QObject::tr(
                                    "[FACETS] Failed to create directories %1")
                                    .arg(outputName));
                    return false;
                }
                bool success = qFacets::ExecuteExportFacets(
                        facets, outputName, params.useNativeOrientation,
                        params.useGlobalOrientation,
                        params.useCustomOrientation, params.nX, params.nY,
                        params.nZ, cmd.silentMode());

                if (!success) {
                    delete group;
                    cmd.error(QObject::tr(
                            "[FACETS] Failed to Export Facets to shape file."));
                    return false;
                }
            }

            if (params.exportFacetsInfo) {
                cmd.print(
                        QObject::tr("[FACETS] Exporting Facets info to csv."));
                if (facets.empty()) {
                    delete group;
                    cmd.error(
                            QObject::tr("[FACETS] Need facets. Must have "
                                        "-EXTRACT_FACETS."));
                    return false;
                }

                QFileInfo fileInfo(params.csvFilename);
                QString directoryPath = fileInfo.path();
                QString newFileName = clCloud.pc->getName() + QString("_") +
                                      fileInfo.fileName();
                QString outputName = QDir(directoryPath).filePath(newFileName);
                QDir dir;
                if (!dir.mkpath(directoryPath)) {
                    delete group;
                    cmd.error(
                            QObject::tr(
                                    "[FACETS] Failed to create directories %1")
                                    .arg(outputName));
                    return false;
                }
                bool success = qFacets::ExecuteExportFacetsInfo(
                        facets, outputName, params.coordsInCsv,
                        params.useNativeOrientation,
                        params.useGlobalOrientation,
                        params.useCustomOrientation, params.nX, params.nY,
                        params.nZ, cmd.silentMode());
                if (!success) {
                    delete group;
                    cmd.error(QObject::tr(
                            "[FACETS] Failed to Export Facets Info to csv"));
                    return false;
                }
            }

            if (group && cmd.autoSaveMode()) {
                CLGroupDesc grpDesc(group,
                                    clCloud.basename + QObject::tr("_FACETS"),
                                    clCloud.path);
                QString errStr = cmd.exportEntity(
                        grpDesc, QString(), nullptr,
                        ccCommandLineInterface::ExportOption::ForceHierarchy);
                if (!errStr.isEmpty()) {
                    cmd.warning(errStr);
                }
            }

            delete group;
        }

        return true;
    }
};
