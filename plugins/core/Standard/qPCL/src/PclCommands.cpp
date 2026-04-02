// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Force full template compilation for PCL features (SHOT, BOARD, etc.)
// whose OMP variants may not be pre-instantiated in the PCL shared library.
#define PCL_NO_PRECOMPILE

#include "PclCommands.h"

#include <Filters/FastGlobalRegistration.h>
#include <PclUtils/PCLModules.h>
#include <PclUtils/cc2sm.h>
#include <PclUtils/sm2cc.h>
#include <ReferenceCloud.h>
#include <ecvCommandLineInterface.h>
#include <ecvGLMatrix.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <cmath>
#include <cstring>
#include <string>

static const char CMD_PCL_SOR[] = "PCL_SOR";
static const char CMD_PCL_NORMAL_EST[] = "PCL_NORMAL_ESTIMATION";
static const char CMD_PCL_MLS[] = "PCL_MLS";
static const char CMD_PCL_EUCLIDEAN[] = "PCL_EUCLIDEAN_CLUSTER";
static const char CMD_PCL_SAC_SEG[] = "PCL_SAC_SEGMENTATION";
static const char CMD_PCL_REGION_GROWING[] = "PCL_REGION_GROWING";
static const char CMD_PCL_MARCHING_CUBES[] = "PCL_MARCHING_CUBES";
static const char CMD_PCL_GREEDY_TRI[] = "PCL_GREEDY_TRIANGULATION";
static const char CMD_PCL_POISSON[] = "PCL_POISSON_RECON";
static const char CMD_PCL_CONVEX_HULL[] = "PCL_CONVEX_HULL";
static const char CMD_PCL_DON[] = "PCL_DON_SEGMENTATION";
static const char CMD_PCL_MINCUT[] = "PCL_MINCUT_SEGMENTATION";
static const char CMD_PCL_FGR[] = "PCL_FAST_GLOBAL_REGISTRATION";
static const char CMD_PCL_SIFT[] = "PCL_EXTRACT_SIFT";
static const char CMD_PCL_PROJ[] = "PCL_PROJECTION_FILTER";
static const char CMD_PCL_GENFILT[] = "PCL_GENERAL_FILTERS";
static const char CMD_PCL_TEMPLATE_ALIGN[] = "PCL_TEMPLATE_ALIGNMENT";
static const char CMD_PCL_CORR_MATCH[] = "PCL_CORRESPONDENCE_MATCHING";

static bool NextFloat(ccCommandLineInterface& cmd, const char* n, float& v) {
    if (cmd.arguments().empty())
        return cmd.error(QObject::tr("Missing value for '-%1'").arg(n));
    bool ok = false;
    v = cmd.arguments().takeFirst().toFloat(&ok);
    return ok ? true : cmd.error(QObject::tr("Invalid value for '-%1'").arg(n));
}

static bool NextInt(ccCommandLineInterface& cmd, const char* n, int& v) {
    if (cmd.arguments().empty())
        return cmd.error(QObject::tr("Missing value for '-%1'").arg(n));
    bool ok = false;
    v = cmd.arguments().takeFirst().toInt(&ok);
    return ok ? true : cmd.error(QObject::tr("Invalid value for '-%1'").arg(n));
}

static bool NeedClouds(ccCommandLineInterface& cmd, const char* t) {
    if (cmd.clouds().empty())
        return cmd.error(
                QObject::tr("No cloud loaded (use -O before -%1)").arg(t));
    return true;
}

namespace {

bool FgrComputeFeatures(ccPointCloud* cloud,
                        fgr::Features& features,
                        double radius) {
    if (!cloud || cloud->size() == 0) return false;
    pcl::PointCloud<pcl::PointNormal>::Ptr tmp_cloud =
            cc2smReader(cloud).getAsPointNormal();
    if (!tmp_cloud) return false;
    pcl::PointCloud<pcl::FPFHSignature33> objectFeatures;
    try {
        pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal,
                               pcl::FPFHSignature33>
                featEstimation;
        featEstimation.setRadiusSearch(radius);
        featEstimation.setInputCloud(tmp_cloud);
        featEstimation.setInputNormals(tmp_cloud);
        featEstimation.compute(objectFeatures);
    } catch (...) {
        return false;
    }
    try {
        unsigned pointCount = cloud->size();
        features.resize(pointCount, Eigen::VectorXf(33));
        for (unsigned i = 0; i < pointCount; ++i) {
            const pcl::FPFHSignature33& feature = objectFeatures.points[i];
            std::memcpy(features[i].data(), feature.histogram,
                        sizeof(float) * 33);
        }
    } catch (const std::bad_alloc&) {
        return false;
    }
    return true;
}

bool FgrCloudToPoints(const ccPointCloud& cloud, fgr::Points& points) {
    unsigned pointCount = cloud.size();
    if (pointCount == 0) return false;
    try {
        points.resize(pointCount);
        for (unsigned i = 0; i < pointCount; ++i) {
            const CCVector3* P = cloud.getPoint(i);
            points[i] = Eigen::Vector3f(P->x, P->y, P->z);
        }
    } catch (const std::bad_alloc&) {
        return false;
    }
    return true;
}

static float DonFieldValue(const PointNT& p, const std::string& field) {
    if (field == "curvature") return p.curvature;
    if (field == "normal_x") return p.normal_x;
    if (field == "normal_y") return p.normal_y;
    if (field == "normal_z") return p.normal_z;
    float nx = p.normal_x, ny = p.normal_y, nz = p.normal_z;
    return std::sqrt(nx * nx + ny * ny + nz * nz);
}

static bool DonPassesRange(const PointNT& p,
                           const std::string& field,
                           float minDon,
                           float maxDon) {
    float v = DonFieldValue(p, field);
    return v >= minDon && v <= maxDon;
}

}  // namespace

// ============================================================================
// Statistical Outlier Removal
// -PCL_SOR [-K 6] [-STD 1.0]
// ============================================================================
struct CmdSOR : public ccCommandLineInterface::Command {
    CmdSOR() : Command("PCL Statistical Outlier Removal", CMD_PCL_SOR) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_SOR)) return false;
        int k = 6;
        float std_t = 1.0f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-K") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "K", k)) return false;
            } else if (a == "-STD") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "STD", std_t)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_SOR] '%1' k=%2 std=%3")
                              .arg(pc->getName())
                              .arg(k)
                              .arg(std_t));
            PCLCloud::Ptr in = cc2smReader(pc).getAsSM();
            if (!in) continue;
            PCLCloud::Ptr out(new PCLCloud);
            if (PCLModules::RemoveOutliersStatistical<PCLCloud>(in, out, k,
                                                                std_t) < 0) {
                cmd.warning(QObject::tr("[PCL_SOR] Failed for '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            ccPointCloud* r = pcl2cc::Convert(*out);
            if (!r) continue;
            r->setName(pc->getName() + "_SOR");
            r->setGlobalScale(pc->getGlobalScale());
            r->setGlobalShift(pc->getGlobalShift());
            d.pc = r;
            d.basename += "_SOR";
            cmd.print(QObject::tr("[PCL_SOR] Result: %1 pts").arg(r->size()));
        }
        return true;
    }
};

// ============================================================================
// Normal Estimation
// -PCL_NORMAL_ESTIMATION [-KNN 10 | -RADIUS 0.5] [-CURVATURE] [-NO_CURVATURE]
// ============================================================================
struct CmdNormalEst : public ccCommandLineInterface::Command {
    CmdNormalEst() : Command("PCL Normal Estimation", CMD_PCL_NORMAL_EST) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_NORMAL_EST)) return false;
        bool useKnn = true;
        float knnR = 10.0f;
        float radius = 0.0f;
        bool curvature = true;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-KNN") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "KNN", knnR)) return false;
                useKnn = true;
            } else if (a == "-RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "RADIUS", radius)) return false;
                useKnn = false;
            } else if (a == "-CURVATURE") {
                cmd.arguments().takeFirst();
                curvature = true;
            } else if (a == "-NO_CURVATURE") {
                cmd.arguments().takeFirst();
                curvature = false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_NORMAL_EST] '%1'").arg(pc->getName()));
            if (useKnn && radius <= 0.0f) {
                ccBBox bb = pc->getOwnBB();
                if (bb.isValid()) radius = bb.getDiagNorm() * 0.005f;
            }
            if (pc->hasNormals()) pc->unallocateNorms();
            PointCloudT::Ptr xyz = cc2smReader(pc).getXYZ2();
            if (!xyz) continue;
            PointCloudNormal::Ptr norms(new PointCloudNormal);
            float p = useKnn ? knnR : radius;
            if (PCLModules::ComputeNormals<PointT, pcl::PointNormal>(
                        xyz, norms, p, useKnn) < 0) {
                cmd.warning(QObject::tr("[PCL_NORMAL_EST] Failed for '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            PCLCloud::Ptr smN(new PCLCloud);
            TO_PCL_CLOUD(*norms, *smN);
            pcl2cc::CopyNormals(*smN, *pc);
            pcl2cc::CopyScalarField(*smN, "curvature", *pc, curvature);
            cmd.print(QObject::tr("[PCL_NORMAL_EST] Done '%1'")
                              .arg(pc->getName()));
        }
        return true;
    }
};

// ============================================================================
// MLS Smoothing
// -PCL_MLS -SEARCH_RADIUS 0.03 [-ORDER 2] [-COMPUTE_NORMALS]
// ============================================================================
struct CmdMLS : public ccCommandLineInterface::Command {
    CmdMLS() : Command("PCL MLS Smoothing", CMD_PCL_MLS) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_MLS)) return false;
        PCLModules::MLSParameters prm;
        prm.search_radius_ = 0.03;
        prm.polynomial_fit_ = true;
        prm.order_ = 2;
        prm.compute_normals_ = false;
        prm.upsample_method_ = PCLModules::MLSParameters::NONE;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            float fv;
            int iv;
            if (a == "-SEARCH_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "SEARCH_RADIUS", fv)) return false;
                prm.search_radius_ = fv;
            } else if (a == "-ORDER") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "ORDER", iv)) return false;
                prm.order_ = iv;
            } else if (a == "-COMPUTE_NORMALS") {
                cmd.arguments().takeFirst();
                prm.compute_normals_ = true;
            } else if (a == "-POLYNOMIAL_FIT") {
                cmd.arguments().takeFirst();
                prm.polynomial_fit_ = true;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_MLS] '%1'").arg(pc->getName()));
            PointCloudT::Ptr xyz = cc2smReader(pc).getXYZ2();
            if (!xyz) continue;
            PointCloudNormal::Ptr out(new PointCloudNormal);
#ifdef LP_PCL_PATCH_ENABLED
            pcl::PointIndicesPtr mapping;
            if (PCLModules::SmoothMls<PointT, pcl::PointNormal>(xyz, prm, out,
                                                                mapping) < 0) {
#else
            if (PCLModules::SmoothMls<PointT, pcl::PointNormal>(xyz, prm, out) <
                0) {
#endif
                cmd.warning(QObject::tr("[PCL_MLS] Failed for '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            PCLCloud::Ptr smOut(new PCLCloud);
            TO_PCL_CLOUD(*out, *smOut);
            ccPointCloud* r = pcl2cc::Convert(*smOut);
            if (!r) continue;
            r->setName(pc->getName() + "_MLS");
            r->setGlobalScale(pc->getGlobalScale());
            r->setGlobalShift(pc->getGlobalShift());
            d.pc = r;
            d.basename += "_MLS";
        }
        return true;
    }
};

// ============================================================================
// Euclidean Cluster Segmentation
// -PCL_EUCLIDEAN_CLUSTER [-TOLERANCE 0.02] [-MIN_SIZE 100] [-MAX_SIZE 250000]
// ============================================================================
struct CmdEuclidean : public ccCommandLineInterface::Command {
    CmdEuclidean() : Command("PCL Euclidean Cluster", CMD_PCL_EUCLIDEAN) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_EUCLIDEAN)) return false;
        float tol = 0.02f;
        int minS = 100, maxS = 250000;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-TOLERANCE") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "TOLERANCE", tol)) return false;
            } else if (a == "-MIN_SIZE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MIN_SIZE", minS)) return false;
            } else if (a == "-MAX_SIZE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MAX_SIZE", maxS)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_EC] '%1'").arg(pc->getName()));
            PointCloudT::Ptr xyz = cc2smReader(pc).getXYZ2();
            if (!xyz) continue;
            std::vector<pcl::PointIndices> ci;
            PCLModules::EuclideanCluster<PointT>(xyz, ci, tol, minS, maxS);
            cmd.print(QObject::tr("[PCL_EC] %1 clusters").arg(ci.size()));
            for (size_t i = 0; i < ci.size(); ++i) {
                cloudViewer::ReferenceCloud sub(pc);
                for (int idx : ci[i].indices)
                    sub.addPointIndex(static_cast<unsigned>(idx));
                ccPointCloud* c = pc->partialClone(&sub);
                if (!c) continue;
                c->setName(QString("%1_cl%2").arg(pc->getName()).arg(i));
                c->setGlobalScale(pc->getGlobalScale());
                c->setGlobalShift(pc->getGlobalShift());
                cmd.clouds().push_back(CLCloudDesc(
                        c, d.basename + QString("_cl%1").arg(i), d.path));
            }
        }
        return true;
    }
};

// ============================================================================
// SAC Segmentation
// -PCL_SAC_SEGMENTATION [-MODEL 0] [-DIST_THRESH 0.01] [-MAX_ITER 1000]
//   [-PROBABILITY 0.95] [-NORMAL_DIST_WEIGHT 0.1]
// ============================================================================
struct CmdSAC : public ccCommandLineInterface::Command {
    CmdSAC() : Command("PCL SAC Segmentation", CMD_PCL_SAC_SEG) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_SAC_SEG)) return false;
        int model = 0, method = 0, maxIter = 100;
        float dist = 0.02f, prob = 0.95f, ndw = 0.1f;
        float minR = -10000.0f, maxR = 10000.0f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-MODEL") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MODEL", model)) return false;
            } else if (a == "-METHOD") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "METHOD", method)) return false;
            } else if (a == "-DIST_THRESH") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "DIST_THRESH", dist)) return false;
            } else if (a == "-MAX_ITER") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MAX_ITER", maxIter)) return false;
            } else if (a == "-PROBABILITY") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "PROBABILITY", prob)) return false;
            } else if (a == "-NORMAL_DIST_WEIGHT") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "NORMAL_DIST_WEIGHT", ndw)) return false;
            } else if (a == "-MIN_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MIN_RADIUS", minR)) return false;
            } else if (a == "-MAX_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MAX_RADIUS", maxR)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_SAC] '%1' model=%2")
                              .arg(pc->getName())
                              .arg(model));
            PointCloudT::Ptr xyz = cc2smReader(pc).getXYZ2();
            if (!xyz) continue;
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
            int r = PCLModules::GetSACSegmentation(xyz, inliers, coeffs, method,
                                                   model, dist, prob, maxIter,
                                                   minR, maxR, ndw);
            if (r < 0 || inliers->indices.empty()) {
                cmd.warning(QObject::tr("[PCL_SAC] No model in '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            cloudViewer::ReferenceCloud sub(pc);
            for (int idx : inliers->indices)
                sub.addPointIndex(static_cast<unsigned>(idx));
            ccPointCloud* seg = pc->partialClone(&sub);
            if (!seg) continue;
            seg->setName(pc->getName() + "_inliers");
            seg->setGlobalScale(pc->getGlobalScale());
            seg->setGlobalShift(pc->getGlobalShift());
            cmd.clouds().push_back(
                    CLCloudDesc(seg, d.basename + "_inliers", d.path));
            cmd.print(QObject::tr("[PCL_SAC] %1/%2 inliers")
                              .arg(inliers->indices.size())
                              .arg(pc->size()));
        }
        return true;
    }
};

// ============================================================================
// Region Growing Segmentation
// -PCL_REGION_GROWING [-SMOOTHNESS 3.0] [-CURVATURE 1.0]
//   [-MIN_SIZE 50] [-MAX_SIZE 100000] [-NEIGHBORS 30]
// ============================================================================
struct CmdRegionGrow : public ccCommandLineInterface::Command {
    CmdRegionGrow() : Command("PCL Region Growing", CMD_PCL_REGION_GROWING) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_REGION_GROWING)) return false;
        int minS = 50, maxS = 100000, nbrs = 30, kSearch = 50;
        float smooth = 3.0f, curv = 1.0f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-SMOOTHNESS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "SMOOTHNESS", smooth)) return false;
            } else if (a == "-CURVATURE") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "CURVATURE", curv)) return false;
            } else if (a == "-MIN_SIZE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MIN_SIZE", minS)) return false;
            } else if (a == "-MAX_SIZE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MAX_SIZE", maxS)) return false;
            } else if (a == "-NEIGHBORS") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "NEIGHBORS", nbrs)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_RG] '%1'").arg(pc->getName()));
            PointCloudT::Ptr xyz = cc2smReader(pc).getXYZ2();
            if (!xyz) continue;
            PointCloudRGB::Ptr colored(new PointCloudRGB);
            std::vector<pcl::PointIndices> clusters;
            int r = PCLModules::GetRegionGrowing(xyz, clusters, colored,
                                                 kSearch, minS, maxS, nbrs,
                                                 smooth, curv);
            if (r < 0 || clusters.empty()) {
                cmd.warning(
                        QObject::tr("[PCL_RG] Failed '%1'").arg(pc->getName()));
                continue;
            }
            cmd.print(QObject::tr("[PCL_RG] %1 regions").arg(clusters.size()));
            for (size_t i = 0; i < clusters.size(); ++i) {
                cloudViewer::ReferenceCloud sub(pc);
                for (int idx : clusters[i].indices)
                    sub.addPointIndex(static_cast<unsigned>(idx));
                ccPointCloud* c = pc->partialClone(&sub);
                if (!c) continue;
                c->setName(QString("%1_rg%2").arg(pc->getName()).arg(i));
                c->setGlobalScale(pc->getGlobalScale());
                c->setGlobalShift(pc->getGlobalShift());
                cmd.clouds().push_back(CLCloudDesc(
                        c, d.basename + QString("_rg%1").arg(i), d.path));
            }
        }
        return true;
    }
};

// ============================================================================
// Greedy Triangulation
// -PCL_GREEDY_TRIANGULATION [-SEARCH_RADIUS 25] [-MAX_NEIGHBORS 100]
//   [-MAX_SURFACE_ANGLE 45] [-MIN_ANGLE 10] [-MAX_ANGLE 120]
// ============================================================================
struct CmdGreedyTri : public ccCommandLineInterface::Command {
    CmdGreedyTri() : Command("PCL Greedy Triangulation", CMD_PCL_GREEDY_TRI) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_GREEDY_TRI)) return false;
        int sR = 25, maxN = 100, maxSA = 45, minA = 10, maxA = 120;
        float wf = 2.5f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-SEARCH_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "SEARCH_RADIUS", sR)) return false;
            } else if (a == "-MAX_NEIGHBORS") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MAX_NEIGHBORS", maxN)) return false;
            } else if (a == "-MAX_SURFACE_ANGLE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MAX_SURFACE_ANGLE", maxSA)) return false;
            } else if (a == "-MIN_ANGLE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MIN_ANGLE", minA)) return false;
            } else if (a == "-MAX_ANGLE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MAX_ANGLE", maxA)) return false;
            } else if (a == "-WEIGHTING") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "WEIGHTING", wf)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_GT] '%1'").arg(pc->getName()));
            pcl::PointCloud<pcl::PointNormal>::Ptr pn =
                    cc2smReader(pc).getAsPointNormal();
            if (!pn || pn->empty()) {
                cmd.warning(QObject::tr("[PCL_GT] '%1' needs normals")
                                    .arg(pc->getName()));
                continue;
            }
            PCLMesh mesh;
            if (PCLModules::GetGreedyTriangulation(pn, mesh, sR, wf, maxN,
                                                   maxSA, minA, maxA) < 0) {
                cmd.warning(
                        QObject::tr("[PCL_GT] Failed '%1'").arg(pc->getName()));
                continue;
            }
            ccMesh* m = pcl2cc::Convert(mesh.cloud, mesh.polygons);
            if (m) {
                m->setName(pc->getName() + "_GT");
                cmd.meshes().push_back(
                        CLMeshDesc(m, d.basename + "_GT", d.path));
            }
        }
        return true;
    }
};

// ============================================================================
// Poisson Surface Reconstruction
// -PCL_POISSON_RECON [-DEPTH 8] [-SCALE 1.25] [-SAMPLES_PER_NODE 3.0]
// ============================================================================
struct CmdPoisson : public ccCommandLineInterface::Command {
    CmdPoisson() : Command("PCL Poisson Reconstruction", CMD_PCL_POISSON) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_POISSON)) return false;
        int depth = 8, isoDiv = 8, solDiv = 8, degree = 2;
        float scale = 1.25f, spn = 3.0f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-DEPTH") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "DEPTH", depth)) return false;
            } else if (a == "-SCALE") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "SCALE", scale)) return false;
            } else if (a == "-SAMPLES_PER_NODE") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "SAMPLES_PER_NODE", spn)) return false;
            } else if (a == "-DEGREE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "DEGREE", degree)) return false;
            } else if (a == "-ISO_DIVIDE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "ISO_DIVIDE", isoDiv)) return false;
            } else if (a == "-SOLVER_DIVIDE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "SOLVER_DIVIDE", solDiv)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_POISSON] '%1'").arg(pc->getName()));
            pcl::PointCloud<pcl::PointNormal>::Ptr pn =
                    cc2smReader(pc).getAsPointNormal();
            if (!pn || pn->empty()) {
                cmd.warning(QObject::tr("[PCL_POISSON] '%1' needs normals")
                                    .arg(pc->getName()));
                continue;
            }
            PCLMesh mesh;
            if (PCLModules::GetPoissonReconstruction(pn, mesh, degree, depth,
                                                     isoDiv, solDiv, scale, spn,
                                                     true, true, false) < 0) {
                cmd.warning(QObject::tr("[PCL_POISSON] Failed '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            ccMesh* m = pcl2cc::Convert(mesh.cloud, mesh.polygons);
            if (m) {
                m->setName(pc->getName() + "_Poisson");
                cmd.meshes().push_back(
                        CLMeshDesc(m, d.basename + "_Poisson", d.path));
            }
        }
        return true;
    }
};

// ============================================================================
// Marching Cubes
// -PCL_MARCHING_CUBES [-METHOD 0] [-GRID_RES 50] [-ISO_LEVEL 0.0]
// ============================================================================
struct CmdMC : public ccCommandLineInterface::Command {
    CmdMC() : Command("PCL Marching Cubes", CMD_PCL_MARCHING_CUBES) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_MARCHING_CUBES)) return false;
        int method = 0, gridRes = 50;
        float iso = 0.0f, eps = 0.01f, pct = 0.0f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-METHOD") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "METHOD", method)) return false;
            } else if (a == "-GRID_RES") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "GRID_RES", gridRes)) return false;
            } else if (a == "-ISO_LEVEL") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "ISO_LEVEL", iso)) return false;
            } else if (a == "-EPSILON") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "EPSILON", eps)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_MC] '%1'").arg(pc->getName()));
            pcl::PointCloud<pcl::PointNormal>::Ptr pn =
                    cc2smReader(pc).getAsPointNormal();
            if (!pn || pn->empty()) {
                cmd.warning(QObject::tr("[PCL_MC] '%1' needs normals")
                                    .arg(pc->getName()));
                continue;
            }
            auto mm = static_cast<PCLModules::MarchingMethod>(method);
            PCLMesh mesh;
            if (PCLModules::GetMarchingCubes<pcl::PointNormal>(
                        pn, mm, mesh, eps, iso, static_cast<float>(gridRes),
                        pct) < 0) {
                cmd.warning(
                        QObject::tr("[PCL_MC] Failed '%1'").arg(pc->getName()));
                continue;
            }
            ccMesh* m = pcl2cc::Convert(mesh.cloud, mesh.polygons);
            if (m) {
                m->setName(pc->getName() + "_MC");
                cmd.meshes().push_back(
                        CLMeshDesc(m, d.basename + "_MC", d.path));
            }
        }
        return true;
    }
};

// ============================================================================
// Convex / Concave Hull
// -PCL_CONVEX_HULL [-ALPHA 0.1] [-DIMENSION 3]
// ============================================================================
struct CmdHull : public ccCommandLineInterface::Command {
    CmdHull() : Command("PCL Convex/Concave Hull", CMD_PCL_CONVEX_HULL) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_CONVEX_HULL)) return false;
        float alpha = 0.0f;
        int dim = 3;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-ALPHA") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "ALPHA", alpha)) return false;
            } else if (a == "-DIMENSION") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "DIMENSION", dim)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_HULL] '%1' alpha=%2")
                              .arg(pc->getName())
                              .arg(alpha));
            PointCloudT::Ptr xyz = cc2smReader(pc).getXYZ2();
            if (!xyz) continue;
            PCLMesh mesh;
            int r;
            if (alpha > 0.0f)
                r = PCLModules::GetConcaveHullReconstruction<PointT>(
                        xyz, mesh, dim, alpha);
            else
                r = PCLModules::GetConvexHullReconstruction<PointT>(xyz, mesh,
                                                                    dim);
            if (r < 0) {
                cmd.warning(QObject::tr("[PCL_HULL] Failed '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            ccMesh* m = pcl2cc::Convert(mesh.cloud, mesh.polygons);
            if (m) {
                m->setName(pc->getName() + "_hull");
                cmd.meshes().push_back(
                        CLMeshDesc(m, d.basename + "_hull", d.path));
            }
        }
        return true;
    }
};

// ============================================================================
// Difference of Normals (DoN) segmentation
// -PCL_DON_SEGMENTATION [-SMALL_SCALE 5] [-LARGE_SCALE 10] [-MIN_DON 0.3]
//   [-MAX_DON 1.3] [-FIELD curvature] [-CLUSTER_TOL 0.02] [-MIN_SIZE 100]
//   [-MAX_SIZE 25000]
// ============================================================================
struct CmdDON : public ccCommandLineInterface::Command {
    CmdDON() : Command("PCL DoN Segmentation", CMD_PCL_DON) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_DON)) return false;
        float smallM = 5.0f, largeM = 10.0f, minDon = 0.3f, maxDon = 1.3f;
        float clusterTolM = 0.02f;
        int minS = 100, maxS = 25000;
        std::string field = "curvature";
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-SMALL_SCALE") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "SMALL_SCALE", smallM)) return false;
            } else if (a == "-LARGE_SCALE") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "LARGE_SCALE", largeM)) return false;
            } else if (a == "-MIN_DON") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MIN_DON", minDon)) return false;
            } else if (a == "-MAX_DON") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MAX_DON", maxDon)) return false;
            } else if (a == "-FIELD") {
                cmd.arguments().takeFirst();
                if (cmd.arguments().empty())
                    return cmd.error(QObject::tr("Missing value for '-FIELD'"));
                field = cmd.arguments().takeFirst().toStdString();
            } else if (a == "-CLUSTER_TOL") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "CLUSTER_TOL", clusterTolM)) return false;
            } else if (a == "-MIN_SIZE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MIN_SIZE", minS)) return false;
            } else if (a == "-MAX_SIZE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MAX_SIZE", maxS)) return false;
            } else
                break;
        }
        if (smallM > largeM || minDon > maxDon || minS > maxS) {
            return cmd.error(QObject::tr("[PCL_DON] Invalid parameter range"));
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_DON] '%1'").arg(pc->getName()));
            PointCloudT::Ptr xyzCloud = cc2smReader(pc).getXYZ2();
            if (!xyzCloud) continue;
            double cloudResolution =
                    PCLModules::ComputeCloudResolution<PointT>(xyzCloud);
            if (cloudResolution <= 0.0) {
                cmd.warning(QObject::tr("[PCL_DON] Bad resolution for '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            float smallR = static_cast<float>(smallM * cloudResolution);
            float largeR = static_cast<float>(largeM * cloudResolution);
            float clusterTol =
                    static_cast<float>(clusterTolM * cloudResolution);
            PointCloudNormal::Ptr normals_small(new PointCloudNormal);
            PointCloudNormal::Ptr normals_large(new PointCloudNormal);
            if (PCLModules::ComputeNormals<PointT, PointNT>(
                        xyzCloud, normals_small, smallR, false, true) < 0 ||
                PCLModules::ComputeNormals<PointT, PointNT>(
                        xyzCloud, normals_large, largeR, false, true) < 0) {
                cmd.warning(
                        QObject::tr("[PCL_DON] Normal estimation failed '%1'")
                                .arg(pc->getName()));
                continue;
            }
            PointCloudNormal::Ptr doncloud(new PointCloudNormal);
            pcl::copyPointCloud<PointT, PointNT>(*xyzCloud, *doncloud);
            if (PCLModules::DONEstimation<PointT, PointNT, PointNT>(
                        xyzCloud, normals_small, normals_large, doncloud) < 0) {
                cmd.warning(QObject::tr("[PCL_DON] DoN failed '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            PointCloudNormal::Ptr don_kept(new PointCloudNormal);
            std::vector<unsigned> orig_idx;
            orig_idx.reserve(doncloud->size());
            for (size_t i = 0; i < doncloud->size(); ++i) {
                if (DonPassesRange(doncloud->points[i], field, minDon,
                                   maxDon)) {
                    orig_idx.push_back(static_cast<unsigned>(i));
                    don_kept->push_back(doncloud->points[i]);
                }
            }
            don_kept->width = static_cast<uint32_t>(don_kept->size());
            don_kept->height = 1;
            if (don_kept->empty()) {
                cmd.warning(QObject::tr("[PCL_DON] No points after magnitude "
                                        "filter '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            std::vector<pcl::PointIndices> cluster_indices;
            PCLModules::EuclideanCluster<PointNT>(don_kept, cluster_indices,
                                                  clusterTol, minS, maxS);
            if (cluster_indices.empty() || cluster_indices.size() > 300) {
                cmd.warning(QObject::tr("[PCL_DON] Clustering failed or too "
                                        "many clusters '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            cmd.print(QObject::tr("[PCL_DON] %1 clusters")
                              .arg(cluster_indices.size()));
            for (size_t ci = 0; ci < cluster_indices.size(); ++ci) {
                cloudViewer::ReferenceCloud sub(pc);
                for (int li : cluster_indices[ci].indices) {
                    if (li < 0 || static_cast<size_t>(li) >= orig_idx.size())
                        continue;
                    sub.addPointIndex(orig_idx[static_cast<size_t>(li)]);
                }
                ccPointCloud* c = pc->partialClone(&sub);
                if (!c) continue;
                c->setName(QString("%1_don%2").arg(pc->getName()).arg(ci));
                c->setGlobalScale(pc->getGlobalScale());
                c->setGlobalShift(pc->getGlobalShift());
                cmd.clouds().push_back(CLCloudDesc(
                        c, d.basename + QString("_don%1").arg(ci), d.path));
            }
        }
        return true;
    }
};

// ============================================================================
// Min-Cut segmentation
// -PCL_MINCUT_SEGMENTATION -FX x -FY y -FZ z [-NEIGHBORS 14] [-SIGMA 0.25]
//   [-BACK_RADIUS 0.8] [-FORE_WEIGHT 0.5]
// ============================================================================
struct CmdMinCut : public ccCommandLineInterface::Command {
    CmdMinCut() : Command("PCL Min-Cut Segmentation", CMD_PCL_MINCUT) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_MINCUT)) return false;
        float fx = 0.f, fy = 0.f, fz = 0.f;
        int neighbours = 14;
        float sigma = 0.25f, backR = 0.8f, foreW = 0.5f;
        bool haveF = false;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-FX") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "FX", fx)) return false;
                haveF = true;
            } else if (a == "-FY") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "FY", fy)) return false;
                haveF = true;
            } else if (a == "-FZ") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "FZ", fz)) return false;
                haveF = true;
            } else if (a == "-NEIGHBORS") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "NEIGHBORS", neighbours)) return false;
            } else if (a == "-SIGMA") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "SIGMA", sigma)) return false;
            } else if (a == "-BACK_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "BACK_RADIUS", backR)) return false;
            } else if (a == "-FORE_WEIGHT") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "FORE_WEIGHT", foreW)) return false;
            } else
                break;
        }
        if (!haveF) {
            return cmd.error(
                    QObject::tr("PCL_MINCUT_SEGMENTATION requires -FX, -FY, "
                                "-FZ (foreground seed)"));
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_MINCUT] '%1'").arg(pc->getName()));
            PCLCloud::Ptr sm_cloud = cc2smReader(pc).getAsSM();
            if (!sm_cloud) continue;
            std::vector<pcl::PointIndices> clusters;
            PointCloudRGB::Ptr cloudSegmented(new PointCloudRGB);
            int result = -1;
            if (pc->hasColors() || pc->hasScalarFields()) {
                PointCloudRGB::Ptr rgbCloud(new PointCloudRGB);
                FROM_PCL_CLOUD(*sm_cloud, *rgbCloud);
                PointRGB fg(0.f, 0.f, 0.f, 255, 255, 255);
                fg.x = fx;
                fg.y = fy;
                fg.z = fz;
                result = PCLModules::GetMinCutSegmentation<PointRGB>(
                        rgbCloud, clusters, cloudSegmented, fg, neighbours,
                        sigma, backR, foreW);
            } else {
                PointCloudT::Ptr xyzCloud(new PointCloudT);
                FROM_PCL_CLOUD(*sm_cloud, *xyzCloud);
                PointT fg(fx, fy, fz);
                result = PCLModules::GetMinCutSegmentation<PointT>(
                        xyzCloud, clusters, cloudSegmented, fg, neighbours,
                        sigma, backR, foreW);
            }
            if (result < 0) {
                cmd.warning(QObject::tr("[PCL_MINCUT] Failed '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            PCLCloud out_sm;
            TO_PCL_CLOUD(*cloudSegmented, out_sm);
            if (out_sm.height * out_sm.width == 0) {
                cmd.warning(QObject::tr("[PCL_MINCUT] Empty result '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            ccPointCloud* r = pcl2cc::Convert(out_sm);
            if (!r) continue;
            r->setName(pc->getName() + "-min-cut");
            r->setGlobalScale(pc->getGlobalScale());
            r->setGlobalShift(pc->getGlobalShift());
            d.pc = r;
            d.basename += "_mincut";
        }
        return true;
    }
};

// ============================================================================
// Fast Global Registration (requires normals; at least two clouds)
// -PCL_FAST_GLOBAL_REGISTRATION -FEATURE_RADIUS r [-REF_INDEX 0]
// ============================================================================
struct CmdFGR : public ccCommandLineInterface::Command {
    CmdFGR() : Command("PCL Fast Global Registration", CMD_PCL_FGR) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (cmd.clouds().size() < 2) {
            return cmd.error(
                    QObject::tr("PCL_FAST_GLOBAL_REGISTRATION needs at least "
                                "two loaded clouds (-O)"));
        }
        float featR = 0.f;
        int refIdx = 0;
        bool haveR = false;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-FEATURE_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "FEATURE_RADIUS", featR)) return false;
                haveR = true;
            } else if (a == "-REF_INDEX") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "REF_INDEX", refIdx)) return false;
            } else
                break;
        }
        if (!haveR || featR <= 0.f) {
            return cmd.error(
                    QObject::tr("PCL_FAST_GLOBAL_REGISTRATION requires "
                                "-FEATURE_RADIUS > 0"));
        }
        if (refIdx < 0 || static_cast<size_t>(refIdx) >= cmd.clouds().size()) {
            return cmd.error(QObject::tr("Invalid -REF_INDEX"));
        }
        ccPointCloud* refPc = cmd.clouds()[static_cast<size_t>(refIdx)].pc;
        if (!refPc || !refPc->hasNormals()) {
            return cmd.error(QObject::tr("Reference cloud must have normals"));
        }
        fgr::Features referenceFeatures;
        if (!FgrComputeFeatures(refPc, referenceFeatures, featR)) {
            return cmd.error(QObject::tr(
                    "FPFH feature computation failed for reference cloud"));
        }
        fgr::Points referencePoints;
        if (!FgrCloudToPoints(*refPc, referencePoints)) {
            return cmd.error(
                    QObject::tr("Failed to read reference cloud points"));
        }
        for (size_t i = 0; i < cmd.clouds().size(); ++i) {
            if (static_cast<int>(i) == refIdx) continue;
            ccPointCloud* aligned = cmd.clouds()[i].pc;
            if (!aligned) continue;
            if (!aligned->hasNormals()) {
                cmd.warning(QObject::tr("[PCL_FGR] Skip '%1' (no normals)")
                                    .arg(aligned->getName()));
                continue;
            }
            cmd.print(QObject::tr("[PCL_FGR] Aligning '%1' to reference '%2'")
                              .arg(aligned->getName())
                              .arg(refPc->getName()));
            fgr::Features alignedFeatures;
            if (!FgrComputeFeatures(aligned, alignedFeatures, featR)) {
                cmd.warning(QObject::tr("[PCL_FGR] FPFH failed for '%1'")
                                    .arg(aligned->getName()));
                continue;
            }
            fgr::Points alignedPoints;
            if (!FgrCloudToPoints(*aligned, alignedPoints)) continue;
            ccGLMatrix ccTrans;
            try {
                fgr::CApp fgrProcess;
                fgrProcess.LoadFeature(referencePoints, referenceFeatures);
                fgrProcess.LoadFeature(alignedPoints, alignedFeatures);
                fgrProcess.NormalizePoints();
                fgrProcess.AdvancedMatching();
                if (!fgrProcess.OptimizePairwise(true)) {
                    cmd.warning(
                            QObject::tr(
                                    "[PCL_FGR] Optimization failed for '%1'")
                                    .arg(aligned->getName()));
                    continue;
                }
                Eigen::Matrix4f trans = fgrProcess.GetOutputTrans();
                for (int k = 0; k < 16; ++k)
                    ccTrans.data()[k] = trans.data()[k];
            } catch (...) {
                cmd.warning(QObject::tr("[PCL_FGR] Exception for '%1'")
                                    .arg(aligned->getName()));
                continue;
            }
            aligned->setGLTransformation(ccTrans);
            cmd.print(QObject::tr("[PCL_FGR] Transformation applied to '%1'")
                              .arg(aligned->getName()));
        }
        return true;
    }
};

// ============================================================================
// SIFT keypoints
// -PCL_EXTRACT_SIFT -MODE RGB|SF -OCTAVES n -MIN_SCALE s -SCALES_PER_OCTAVE m
//   [-FIELD name] [-MIN_CONTRAST c]
// ============================================================================
struct CmdSIFT : public ccCommandLineInterface::Command {
    CmdSIFT() : Command("PCL Extract SIFT", CMD_PCL_SIFT) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_SIFT)) return false;
        int oct = 0, sPerOct = 0;
        float minScale = 0.f, minContrast = 0.f;
        QString modeStr;
        QString fieldName;
        bool useMinContrast = false;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-MODE") {
                cmd.arguments().takeFirst();
                if (cmd.arguments().empty())
                    return cmd.error(QObject::tr("Missing value for '-MODE'"));
                modeStr = cmd.arguments().takeFirst();
            } else if (a == "-OCTAVES") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "OCTAVES", oct)) return false;
            } else if (a == "-MIN_SCALE") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MIN_SCALE", minScale)) return false;
            } else if (a == "-SCALES_PER_OCTAVE") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "SCALES_PER_OCTAVE", sPerOct)) return false;
            } else if (a == "-FIELD") {
                cmd.arguments().takeFirst();
                if (cmd.arguments().empty())
                    return cmd.error(QObject::tr("Missing value for '-FIELD'"));
                fieldName = cmd.arguments().takeFirst();
            } else if (a == "-MIN_CONTRAST") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MIN_CONTRAST", minContrast)) return false;
                useMinContrast = true;
            } else
                break;
        }
        if (!modeStr.size() || oct <= 0 || minScale <= 0.f || sPerOct <= 0) {
            return cmd.error(
                    QObject::tr("PCL_EXTRACT_SIFT requires -MODE RGB|SF, "
                                "-OCTAVES, -MIN_SCALE, -SCALES_PER_OCTAVE (all "
                                "positive where applicable)"));
        }
        if (useMinContrast && minContrast <= 0.f) {
            return cmd.error(
                    QObject::tr("When using -MIN_CONTRAST, value must be > 0"));
        }
        const bool useRgb = modeStr.compare(QStringLiteral("RGB"),
                                            Qt::CaseInsensitive) == 0;
        const bool useSf =
                modeStr.compare(QStringLiteral("SF"), Qt::CaseInsensitive) == 0;
        if (!useRgb && !useSf) {
            return cmd.error(QObject::tr("-MODE must be RGB or SF"));
        }
        if (useSf && fieldName.isEmpty()) {
            return cmd.error(QObject::tr(
                    "-MODE SF requires -FIELD (scalar field name)"));
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_SIFT] '%1'").arg(pc->getName()));
            if (useRgb && !pc->hasColors()) {
                cmd.warning(QObject::tr("[PCL_SIFT] '%1' has no RGB")
                                    .arg(pc->getName()));
                continue;
            }
            std::list<std::string> req_fields;
            req_fields.push_back("xyz");
            if (useRgb)
                req_fields.push_back("rgb");
            else
                req_fields.push_back(qPrintable(fieldName));
            PCLCloud::Ptr sm_cloud = cc2smReader(pc).getAsSM(req_fields);
            if (!sm_cloud) {
                cmd.warning(
                        QObject::tr(
                                "[PCL_SIFT] Cannot build PCL cloud for '%1'")
                                .arg(pc->getName()));
                continue;
            }
            QString fn_ns = fieldName;
            fn_ns.replace(' ', '_');
            std::string field_std = fn_ns.toStdString();
            if (useSf) {
                int field_index = pcl::getFieldIndex(*sm_cloud, field_std);
                if (field_index < 0) {
                    cmd.warning(
                            QObject::tr(
                                    "[PCL_SIFT] Scalar field not found on '%1'")
                                    .arg(pc->getName()));
                    continue;
                }
                sm_cloud->fields[static_cast<size_t>(field_index)].name =
                        "intensity";
            }
            pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(
                    new pcl::PointCloud<pcl::PointXYZ>);
            float mc = (useMinContrast ? minContrast : 0.f);
            if (useSf) {
                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_i(
                        new pcl::PointCloud<pcl::PointXYZI>);
                FROM_PCL_CLOUD(*sm_cloud, *cloud_i);
                PCLModules::EstimateSIFT<pcl::PointXYZI, pcl::PointXYZ>(
                        cloud_i, out_cloud, oct, minScale, sPerOct, mc);
            } else {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(
                        new pcl::PointCloud<pcl::PointXYZRGB>);
                FROM_PCL_CLOUD(*sm_cloud, *cloud_rgb);
                PCLModules::EstimateSIFT<pcl::PointXYZRGB, pcl::PointXYZ>(
                        cloud_rgb, out_cloud, oct, minScale, sPerOct, mc);
            }
            PCLCloud out_sm;
            TO_PCL_CLOUD(*out_cloud, out_sm);
            if (out_sm.height * out_sm.width == 0) {
                cmd.warning(QObject::tr("[PCL_SIFT] Empty keypoints for '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            ccPointCloud* r = pcl2cc::Convert(out_sm);
            if (!r) continue;
            r->setName(pc->getName() + "_SIFT");
            r->setGlobalScale(pc->getGlobalScale());
            r->setGlobalShift(pc->getGlobalShift());
            d.pc = r;
            d.basename += "_SIFT";
            cmd.print(QObject::tr("[PCL_SIFT] %1 keypoints").arg(r->size()));
        }
        return true;
    }
};

// ============================================================================
// Project inliers to a plane Ax+By+Cz+D=0
// -PCL_PROJECTION_FILTER -A a -B b -C c -D d
// ============================================================================
struct CmdProj : public ccCommandLineInterface::Command {
    CmdProj() : Command("PCL Projection Filter", CMD_PCL_PROJ) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_PROJ)) return false;
        float A = 0.f, B = 0.f, C = 1.f, D = 0.f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-A") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "A", A)) return false;
            } else if (a == "-B") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "B", B)) return false;
            } else if (a == "-C") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "C", C)) return false;
            } else if (a == "-D") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "D", D)) return false;
            } else
                break;
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_PROJ] '%1'").arg(pc->getName()));
            PointCloudT::Ptr xyzCloud = cc2smReader(pc).getXYZ2();
            if (!xyzCloud) continue;
            PointCloudT::Ptr outXyz(new PointCloudT);
            pcl::ModelCoefficients::Ptr coefficients(
                    new pcl::ModelCoefficients());
            coefficients->values.resize(4);
            coefficients->values[0] = A;
            coefficients->values[1] = B;
            coefficients->values[2] = C;
            coefficients->values[3] = D;
            if (PCLModules::GetProjection(xyzCloud, outXyz, coefficients, 0) <
                0) {
                cmd.warning(QObject::tr("[PCL_PROJ] Failed '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            PCLCloud::Ptr out_sm(new PCLCloud);
            TO_PCL_CLOUD(*outXyz, *out_sm);
            ccPointCloud* r = pcl2cc::Convert(*out_sm);
            if (!r) continue;
            r->setName(pc->getName() + "-projection");
            r->setGlobalScale(pc->getGlobalScale());
            r->setGlobalShift(pc->getGlobalShift());
            d.pc = r;
            d.basename += "_projection";
        }
        return true;
    }
};

// ============================================================================
// General filters: PassThrough or VoxelGrid
// -PCL_GENERAL_FILTERS -MODE PASS|VOXEL [-FIELD z] [-MIN v] [-MAX v]
//   [-LEAF ...] [-LEAF_X] [-LEAF_Y] [-LEAF_Z]
// ============================================================================
struct CmdGenFilt : public ccCommandLineInterface::Command {
    CmdGenFilt() : Command("PCL General Filters", CMD_PCL_GENFILT) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (!NeedClouds(cmd, CMD_PCL_GENFILT)) return false;
        QString mode;
        QString ptField = QStringLiteral("z");
        float ptMin = 0.1f, ptMax = 1.1f;
        float lx = -1.f, ly = -1.f, lz = -1.f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-MODE") {
                cmd.arguments().takeFirst();
                if (cmd.arguments().empty())
                    return cmd.error(QObject::tr("Missing value for '-MODE'"));
                mode = cmd.arguments().takeFirst();
            } else if (a == "-FIELD") {
                cmd.arguments().takeFirst();
                if (cmd.arguments().empty())
                    return cmd.error(QObject::tr("Missing value for '-FIELD'"));
                ptField = cmd.arguments().takeFirst();
            } else if (a == "-MIN") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MIN", ptMin)) return false;
            } else if (a == "-MAX") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MAX", ptMax)) return false;
            } else if (a == "-LEAF") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "LEAF", lx)) return false;
                ly = lz = lx;
            } else if (a == "-LEAF_X") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "LEAF_X", lx)) return false;
            } else if (a == "-LEAF_Y") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "LEAF_Y", ly)) return false;
            } else if (a == "-LEAF_Z") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "LEAF_Z", lz)) return false;
            } else
                break;
        }
        if (!mode.size()) {
            return cmd.error(QObject::tr(
                    "PCL_GENERAL_FILTERS requires -MODE PASS or VOXEL"));
        }
        const bool pass =
                mode.compare(QStringLiteral("PASS"), Qt::CaseInsensitive) == 0;
        const bool voxel =
                mode.compare(QStringLiteral("VOXEL"), Qt::CaseInsensitive) == 0;
        if (!pass && !voxel) {
            return cmd.error(QObject::tr("-MODE must be PASS or VOXEL"));
        }
        for (CLCloudDesc& d : cmd.clouds()) {
            ccPointCloud* pc = d.pc;
            if (!pc) continue;
            cmd.print(QObject::tr("[PCL_GEN] '%1' mode=%2")
                              .arg(pc->getName())
                              .arg(mode));
            PCLCloud::Ptr sm_cloud = cc2smReader(pc).getAsSM();
            if (!sm_cloud) continue;
            PCLCloud::Ptr out_sm(new PCLCloud);
            const bool hasColorOrSf = pc->hasColors() || pc->hasScalarFields();
            if (pass) {
                if (hasColorOrSf) {
                    PointCloudRGB::Ptr rgbIn(new PointCloudRGB);
                    PointCloudRGB::Ptr rgbOut(new PointCloudRGB);
                    FROM_PCL_CLOUD(*sm_cloud, *rgbIn);
                    if (PCLModules::PassThroughFilter<PointRGB>(
                                rgbIn, rgbOut, ptField, ptMin, ptMax) < 0) {
                        cmd.warning(
                                QObject::tr("[PCL_GEN] PassThrough failed '%1'")
                                        .arg(pc->getName()));
                        continue;
                    }
                    TO_PCL_CLOUD(*rgbOut, *out_sm);
                } else {
                    PointCloudT::Ptr xyzIn(new PointCloudT);
                    PointCloudT::Ptr xyzOut(new PointCloudT);
                    FROM_PCL_CLOUD(*sm_cloud, *xyzIn);
                    if (PCLModules::PassThroughFilter<PointT>(
                                xyzIn, xyzOut, ptField, ptMin, ptMax) < 0) {
                        cmd.warning(
                                QObject::tr("[PCL_GEN] PassThrough failed '%1'")
                                        .arg(pc->getName()));
                        continue;
                    }
                    TO_PCL_CLOUD(*xyzOut, *out_sm);
                }
            } else {
                if (lx < 0.f && ly < 0.f && lz < 0.f) {
                    lx = ly = lz = -1.f;
                } else if (lx >= 0.f && ly < 0.f && lz < 0.f) {
                    ly = lz = lx;
                }
                if (hasColorOrSf) {
                    PointCloudRGB::Ptr rgbIn(new PointCloudRGB);
                    PointCloudRGB::Ptr rgbOut(new PointCloudRGB);
                    FROM_PCL_CLOUD(*sm_cloud, *rgbIn);
                    if (PCLModules::VoxelGridFilter<PointRGB>(rgbIn, rgbOut, lx,
                                                              ly, lz) < 0) {
                        cmd.warning(QObject::tr("[PCL_GEN] Voxel failed '%1'")
                                            .arg(pc->getName()));
                        continue;
                    }
                    if (rgbOut->size() == rgbIn->size()) {
                        cmd.warning(QObject::tr("[PCL_GEN] Voxel had no effect "
                                                "(leaf too small?) '%1'")
                                            .arg(pc->getName()));
                        continue;
                    }
                    TO_PCL_CLOUD(*rgbOut, *out_sm);
                } else {
                    PointCloudT::Ptr xyzIn(new PointCloudT);
                    PointCloudT::Ptr xyzOut(new PointCloudT);
                    FROM_PCL_CLOUD(*sm_cloud, *xyzIn);
                    if (PCLModules::VoxelGridFilter<PointT>(xyzIn, xyzOut, lx,
                                                            ly, lz) < 0) {
                        cmd.warning(QObject::tr("[PCL_GEN] Voxel failed '%1'")
                                            .arg(pc->getName()));
                        continue;
                    }
                    if (xyzOut->size() == xyzIn->size()) {
                        cmd.warning(QObject::tr("[PCL_GEN] Voxel had no effect "
                                                "(leaf too small?) '%1'")
                                            .arg(pc->getName()));
                        continue;
                    }
                    TO_PCL_CLOUD(*xyzOut, *out_sm);
                }
            }
            if (out_sm->width * out_sm->height == 0) {
                cmd.warning(QObject::tr("[PCL_GEN] Empty result '%1'")
                                    .arg(pc->getName()));
                continue;
            }
            ccPointCloud* r = pcl2cc::Convert(*out_sm);
            if (!r) continue;
            r->setName(pass ? QString(pc->getName() + "-passThrough")
                            : QString(pc->getName() + "-voxelGrid"));
            r->setGlobalScale(pc->getGlobalScale());
            r->setGlobalShift(pc->getGlobalShift());
            d.pc = r;
            d.basename += pass ? "_pass" : "_voxel";
        }
        return true;
    }
};

// ============================================================================
// Template Alignment (SAC-IA with FPFH)
// -PCL_TEMPLATE_ALIGNMENT -NORMAL_RADIUS r -FEATURE_RADIUS r
//   [-MAX_ITERATIONS 500] [-MIN_SAMPLE_DIST 0.05] [-MAX_CORR_DIST 0.01]
//   [-VOXEL_LEAF 0.005] [-REF_INDEX 0]
// Requires at least 2 loaded clouds.
// ============================================================================
struct CmdTemplateAlign : public ccCommandLineInterface::Command {
    CmdTemplateAlign()
        : Command("PCL Template Alignment", CMD_PCL_TEMPLATE_ALIGN) {}
    //! Executes the PCL_TEMPLATE_ALIGNMENT command.
    /** Parses command-line arguments, selects the reference/target clouds,
        and runs SAC-IA template alignment using FPFH features. The last
        loaded cloud (or the one specified by -REF_INDEX) is used as target. */
    bool process(ccCommandLineInterface& cmd) override {
        // At least two clouds are required: one or more templates and one target.
        if (cmd.clouds().size() < 2) {
            return cmd.error(QObject::tr(
                    "PCL_TEMPLATE_ALIGNMENT needs >=2 clouds (templates + "
                    "target). Last cloud = target."));
        }
        // Default parameters for normal estimation, feature computation and SAC-IA.
        float normalR = 0.02f, featureR = 0.02f;
        int maxIter = 500;
        float minSampleDist = 0.05f, maxCorrDist = 0.01f;
        float voxelLeaf = -1.f;
        int refIdx = -1;
        // Parse optional command-line arguments overriding the default parameters.
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-NORMAL_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "NORMAL_RADIUS", normalR)) return false;
            } else if (a == "-FEATURE_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "FEATURE_RADIUS", featureR)) return false;
            } else if (a == "-MAX_ITERATIONS") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "MAX_ITERATIONS", maxIter)) return false;
            } else if (a == "-MIN_SAMPLE_DIST") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MIN_SAMPLE_DIST", minSampleDist))
                    return false;
            } else if (a == "-MAX_CORR_DIST") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MAX_CORR_DIST", maxCorrDist)) return false;
            } else if (a == "-VOXEL_LEAF") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "VOXEL_LEAF", voxelLeaf)) return false;
            } else if (a == "-REF_INDEX") {
                cmd.arguments().takeFirst();
                if (!NextInt(cmd, "REF_INDEX", refIdx)) return false;
            } else
                break;
        }
        // If no explicit reference index is provided, use the last cloud as target.
        size_t targetIdx = (refIdx >= 0) ? static_cast<size_t>(refIdx)
                                         : cmd.clouds().size() - 1;
        if (targetIdx >= cmd.clouds().size()) {
            return cmd.error(QObject::tr("Invalid -REF_INDEX"));
        }
        ccPointCloud* targetPc = cmd.clouds()[targetIdx].pc;
        if (!targetPc) return cmd.error(QObject::tr("Target cloud is null"));
        cmd.print(QObject::tr("[PCL_TMPL] Target: '%1'")
                          .arg(targetPc->getName()));
        PointCloudT::Ptr targetXyz = cc2smReader(targetPc).getXYZ2();
        if (!targetXyz)
            return cmd.error(QObject::tr("Cannot convert target cloud"));
        if (voxelLeaf > 0.f) {
            PointCloudT::Ptr downsampled(new PointCloudT);
            PCLModules::VoxelGridFilter<PointT>(targetXyz, downsampled,
                                                voxelLeaf);
            targetXyz = downsampled;
        }
        PCLModules::FeatureCloud targetFC;
        targetFC.setNormalRadius(normalR);
        targetFC.setFeatureRadius(featureR);
        targetFC.setInputCloud(targetXyz);
        PCLModules::TemplateMatching matcher;
        matcher.setmaxIterations(maxIter);
        matcher.setminSampleDis(minSampleDist);
        matcher.setmaxCorrespondenceDis(maxCorrDist * maxCorrDist);
        matcher.setTargetCloud(targetFC);
        for (size_t i = 0; i < cmd.clouds().size(); ++i) {
            if (i == targetIdx) continue;
            ccPointCloud* tmplPc = cmd.clouds()[i].pc;
            if (!tmplPc) continue;
            PointCloudT::Ptr tmplXyz = cc2smReader(tmplPc).getXYZ2();
            if (!tmplXyz) continue;
            PCLModules::FeatureCloud tmplFC;
            tmplFC.setNormalRadius(normalR);
            tmplFC.setFeatureRadius(featureR);
            tmplFC.setInputCloud(tmplXyz);
            matcher.addTemplateCloud(tmplFC);
        }
        PCLModules::TemplateMatching::Result best;
        int bestIdx = matcher.findBestAlignment(best);
        if (bestIdx < 0) {
            return cmd.error(QObject::tr("[PCL_TMPL] Alignment failed"));
        }
        cmd.print(QObject::tr("[PCL_TMPL] Best template index=%1 score=%2")
                          .arg(bestIdx)
                          .arg(best.fitness_score));
        size_t srcI = 0;
        for (size_t i = 0; i < cmd.clouds().size(); ++i) {
            if (i == targetIdx) continue;
            if (static_cast<int>(srcI) == bestIdx) {
                ccGLMatrix ccTrans;
                for (int k = 0; k < 16; ++k)
                    ccTrans.data()[k] = best.final_transformation.data()[k];
                cmd.clouds()[i].pc->setGLTransformation(ccTrans);
                cmd.print(QObject::tr("[PCL_TMPL] Applied to '%1'")
                                  .arg(cmd.clouds()[i].pc->getName()));
                break;
            }
            ++srcI;
        }
        return true;
    }
};

// ============================================================================
// Correspondence Matching (GC or Hough3D grouping)
// -PCL_CORRESPONDENCE_MATCHING -MODEL_RADIUS r -SCENE_RADIUS r
//   [-SHOT_RADIUS 0.03] [-NORMAL_K 10] [-GC] [-HOUGH]
//   [-GC_RESOLUTION 0.01] [-GC_MIN_CLUSTER 20] [-HOUGH_BIN 0.01]
//   [-HOUGH_THRESHOLD 5] [-HOUGH_LRF 0.015] [-VOXEL_LEAF 0.005]
// Requires 2+ loaded clouds. Last cloud = scene, others = models.
// ============================================================================
struct CmdCorrMatch : public ccCommandLineInterface::Command {
    CmdCorrMatch()
        : Command("PCL Correspondence Matching", CMD_PCL_CORR_MATCH) {}
    bool process(ccCommandLineInterface& cmd) override {
        if (cmd.clouds().size() < 2) {
            return cmd.error(QObject::tr(
                    "PCL_CORRESPONDENCE_MATCHING needs >=2 clouds (model(s) "
                    "+ scene). Last cloud = scene."));
        }
        float modelR = 0.02f, sceneR = 0.03f, shotR = 0.03f;
        float normalK = 10.f;
        bool gcMode = true;
        float gcRes = 0.01f, gcMinCluster = 20.f;
        float lrfR = 0.015f, houghBin = 0.01f, houghThresh = 5.f;
        float voxelLeaf = -1.f;
        while (!cmd.arguments().empty()) {
            QString a = cmd.arguments().front().toUpper();
            if (a == "-MODEL_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "MODEL_RADIUS", modelR)) return false;
            } else if (a == "-SCENE_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "SCENE_RADIUS", sceneR)) return false;
            } else if (a == "-SHOT_RADIUS") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "SHOT_RADIUS", shotR)) return false;
            } else if (a == "-NORMAL_K") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "NORMAL_K", normalK)) return false;
            } else if (a == "-GC") {
                cmd.arguments().takeFirst();
                gcMode = true;
            } else if (a == "-HOUGH") {
                cmd.arguments().takeFirst();
                gcMode = false;
            } else if (a == "-GC_RESOLUTION") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "GC_RESOLUTION", gcRes)) return false;
            } else if (a == "-GC_MIN_CLUSTER") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "GC_MIN_CLUSTER", gcMinCluster))
                    return false;
            } else if (a == "-HOUGH_BIN") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "HOUGH_BIN", houghBin)) return false;
            } else if (a == "-HOUGH_THRESHOLD") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "HOUGH_THRESHOLD", houghThresh))
                    return false;
            } else if (a == "-HOUGH_LRF") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "HOUGH_LRF", lrfR)) return false;
            } else if (a == "-VOXEL_LEAF") {
                cmd.arguments().takeFirst();
                if (!NextFloat(cmd, "VOXEL_LEAF", voxelLeaf)) return false;
            } else
                break;
        }
        size_t sceneIdx = cmd.clouds().size() - 1;
        ccPointCloud* scenePc = cmd.clouds()[sceneIdx].pc;
        if (!scenePc) return cmd.error(QObject::tr("Scene cloud is null"));
        PCLCloud::Ptr sceneSm = cc2smReader(scenePc).getAsSM();
        if (!sceneSm)
            return cmd.error(QObject::tr("Cannot convert scene cloud"));
        PointCloudT::Ptr sceneXyz(new PointCloudT);
        FROM_PCL_CLOUD(*sceneSm, *sceneXyz);
        if (voxelLeaf > 0.f) {
            PointCloudT::Ptr ds(new PointCloudT);
            PCLModules::VoxelGridFilter<PointT>(sceneXyz, ds, voxelLeaf);
            sceneXyz = ds;
        }
        PointCloudNormal::Ptr sceneNormals(new PointCloudNormal);
        PCLModules::ComputeNormals<PointT, PointNT>(sceneXyz, sceneNormals, 0.f,
                                                    false, false,
                                                    static_cast<int>(normalK));
        PointCloudT::Ptr sceneKeypoints(new PointCloudT);
        PCLModules::GetUniformSampling<PointT>(sceneXyz, sceneKeypoints,
                                               sceneR);
        pcl::PointCloud<pcl::SHOT352>::Ptr sceneDescriptors(
                new pcl::PointCloud<pcl::SHOT352>);
        PCLModules::EstimateShot<PointT, PointNT, pcl::SHOT352>(
                sceneXyz, sceneKeypoints, sceneNormals, sceneDescriptors,
                shotR);
        for (size_t mi = 0; mi < sceneIdx; ++mi) {
            ccPointCloud* modelPc = cmd.clouds()[mi].pc;
            if (!modelPc) continue;
            cmd.print(QObject::tr("[PCL_CORR] Model '%1' → Scene '%2'")
                              .arg(modelPc->getName())
                              .arg(scenePc->getName()));
            PCLCloud::Ptr modelSm = cc2smReader(modelPc).getAsSM();
            if (!modelSm) continue;
            PointCloudT::Ptr modelXyz(new PointCloudT);
            FROM_PCL_CLOUD(*modelSm, *modelXyz);
            PointCloudNormal::Ptr modelNormals(new PointCloudNormal);
            PCLModules::ComputeNormals<PointT, PointNT>(
                    modelXyz, modelNormals, 0.f, false, false,
                    static_cast<int>(normalK));
            PointCloudT::Ptr modelKeypoints(new PointCloudT);
            PCLModules::GetUniformSampling<PointT>(modelXyz, modelKeypoints,
                                                   modelR);
            pcl::PointCloud<pcl::SHOT352>::Ptr modelDescriptors(
                    new pcl::PointCloud<pcl::SHOT352>);
            PCLModules::EstimateShot<PointT, PointNT, pcl::SHOT352>(
                    modelXyz, modelKeypoints, modelNormals, modelDescriptors,
                    shotR);
            pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
            pcl::KdTreeFLANN<pcl::SHOT352> matchSearch;
            matchSearch.setInputCloud(modelDescriptors);
            for (size_t si = 0; si < sceneDescriptors->size(); ++si) {
                std::vector<int> nn_indices(1);
                std::vector<float> nn_sqr_dists(1);
                if (!std::isfinite(sceneDescriptors->at(si).descriptor[0]))
                    continue;
                int found = matchSearch.nearestKSearch(
                        sceneDescriptors->at(si), 1, nn_indices, nn_sqr_dists);
                if (found == 1 && nn_sqr_dists[0] < 0.25f) {
                    pcl::Correspondence c(nn_indices[0], static_cast<int>(si),
                                          nn_sqr_dists[0]);
                    corrs->push_back(c);
                }
            }
            cmd.print(QObject::tr("[PCL_CORR] %1 correspondences")
                              .arg(corrs->size()));
            std::vector<Eigen::Matrix4f,
                        Eigen::aligned_allocator<Eigen::Matrix4f>>
                    rotoTranslations;
            std::vector<pcl::Correspondences> clusteredCorrs;
            if (gcMode) {
                PCLModules::EstimateGeometricConsistencyGrouping<PointT,
                                                                 PointT>(
                        modelKeypoints, sceneKeypoints, corrs, rotoTranslations,
                        clusteredCorrs, gcRes, static_cast<int>(gcMinCluster));
            } else {
                pcl::PointCloud<pcl::ReferenceFrame>::Ptr modelRF(
                        new pcl::PointCloud<pcl::ReferenceFrame>);
                pcl::PointCloud<pcl::ReferenceFrame>::Ptr sceneRF(
                        new pcl::PointCloud<pcl::ReferenceFrame>);
                PCLModules::EstimateLocalReferenceFrame<PointT, PointNT>(
                        modelXyz, modelKeypoints, modelNormals, modelRF, lrfR);
                PCLModules::EstimateLocalReferenceFrame<PointT, PointNT>(
                        sceneXyz, sceneKeypoints, sceneNormals, sceneRF, lrfR);
                PCLModules::EstimateHough3DGrouping<PointT, PointT>(
                        modelKeypoints, sceneKeypoints, modelRF, sceneRF, corrs,
                        rotoTranslations, clusteredCorrs, houghBin,
                        houghThresh);
            }
            cmd.print(QObject::tr("[PCL_CORR] %1 instances found")
                              .arg(rotoTranslations.size()));
            for (size_t ii = 0; ii < rotoTranslations.size(); ++ii) {
                PointCloudT::Ptr transformed(new PointCloudT);
                pcl::transformPointCloud(*modelXyz, *transformed,
                                         rotoTranslations[ii]);
                PCLCloud out_sm;
                TO_PCL_CLOUD(*transformed, out_sm);
                ccPointCloud* r = pcl2cc::Convert(out_sm);
                if (!r) continue;
                r->setName(QString("%1_instance%2")
                                   .arg(modelPc->getName())
                                   .arg(ii));
                r->setGlobalScale(modelPc->getGlobalScale());
                r->setGlobalShift(modelPc->getGlobalShift());
                cmd.clouds().push_back(CLCloudDesc(
                        r,
                        cmd.clouds()[mi].basename + QString("_inst%1").arg(ii),
                        cmd.clouds()[mi].path));
            }
        }
        return true;
    }
};

// ============================================================================
void PclCommands::RegisterAll(ccCommandLineInterface* cmd) {
    if (!cmd) return;
    using S = ccCommandLineInterface::Command::Shared;
    cmd->registerCommand(S(new CmdSOR));
    cmd->registerCommand(S(new CmdNormalEst));
    cmd->registerCommand(S(new CmdMLS));
    cmd->registerCommand(S(new CmdEuclidean));
    cmd->registerCommand(S(new CmdSAC));
    cmd->registerCommand(S(new CmdRegionGrow));
    cmd->registerCommand(S(new CmdGreedyTri));
    cmd->registerCommand(S(new CmdPoisson));
    cmd->registerCommand(S(new CmdMC));
    cmd->registerCommand(S(new CmdHull));
    cmd->registerCommand(S(new CmdDON));
    cmd->registerCommand(S(new CmdMinCut));
    cmd->registerCommand(S(new CmdFGR));
    cmd->registerCommand(S(new CmdSIFT));
    cmd->registerCommand(S(new CmdProj));
    cmd->registerCommand(S(new CmdGenFilt));
    cmd->registerCommand(S(new CmdTemplateAlign));
    cmd->registerCommand(S(new CmdCorrMatch));
}
