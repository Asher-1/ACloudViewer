//##########################################################################
//#                                                                        #
//#                       CLOUDCOMPARE PLUGIN: qPCL                        #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#                    COPYRIGHT: CloudCompare project                     #
//#                                                                        #
//##########################################################################

#include "FastGlobalRegistrationFilter.h"

#include "FastGlobalRegistration.h"
#include "dialogs/FastGlobalRegistrationDlg.h"

// Local
#include "PclUtils/PCLModules.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"

// PCL
#include <pcl/features/fpfh_omp.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>

// Qt
#include <QMainWindow>

// Boost
#include <boost/make_shared.hpp>

// Error codes
static constexpr int NoNormals = -11;

FastGlobalRegistrationFilter::FastGlobalRegistrationFilter()
    : BasePclModule(PclModuleDescription(
              tr("Fast Global Registration"),
              tr("Fast Global Registration, by Zhou et al."),
              tr("Attempts to automatically register clouds (with normals) "
                 "without initial alignment"),
              ":/toolbar/PclAlgorithms/icons/fastGlobalRegistration.png")),
      m_alignedClouds(),
      m_referenceCloud(nullptr),
      m_featureRadius(0) {}

FastGlobalRegistrationFilter::~FastGlobalRegistrationFilter() {}

QString FastGlobalRegistrationFilter::getErrorMessage(int errorCode) {
    switch (errorCode) {
        case NoNormals:
            return tr("Clouds must have normals");
        default:
            // see below
            break;
    }

    return BasePclModule::getErrorMessage(errorCode);
}

int FastGlobalRegistrationFilter::checkSelected() {
    if (m_selected.size() < 2) {
        return -11;
    }

    for (ccHObject* entity : m_selected) {
        // clouds only
        if (!entity->isA(CV_TYPES::POINT_CLOUD)) return -11;
    }

    return 1;
}

static bool ComputeFeatures(ccPointCloud* cloud,
                            fgr::Features& features,
                            double radius) {
    if (!cloud) {
        assert(false);
        return false;
    }

    unsigned pointCount = cloud->size();
    if (pointCount == 0) {
        CVLog::Warning("Cloud is empty");
        return false;
    }

    pcl::PointCloud<pcl::PointNormal>::Ptr tmp_cloud = cc2smReader(cloud).getAsPointNormal();
    if (!tmp_cloud) {
        CVLog::Warning("Failed to convert CC cloud to PCL cloud");
        return false;
    }

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
        CVLog::Warning("Failed to compute FPFH feature descriptors");
        return false;
    }

    try {
        features.resize(pointCount, Eigen::VectorXf(33));
        for (unsigned i = 0; i < pointCount; ++i) {
            const pcl::FPFHSignature33& feature = objectFeatures.points[i];
            memcpy(features[i].data(), feature.histogram, sizeof(float) * 33);
        }
    } catch (const std::bad_alloc&) {
        CVLog::Warning("Not enough memory");
        return false;
    }

    return true;
}

static bool ConverFromTo(const ccPointCloud& cloud, fgr::Points& points) {
    unsigned pointCount = cloud.size();
    if (pointCount == 0) {
        CVLog::Warning("Cloud is empty");
        return false;
    }

    try {
        points.resize(pointCount);
        for (unsigned i = 0; i < pointCount; ++i) {
            const CCVector3* P = cloud.getPoint(i);
            points[i] = Eigen::Vector3f(P->x, P->y, P->z);
        }
    } catch (const std::bad_alloc&) {
        CVLog::Warning("Not enough memory");
        return false;
    }

    return true;
}

void FastGlobalRegistrationFilter::getParametersFromDialog() {
    // just in case
    m_alignedClouds.clear();
    m_referenceCloud = nullptr;

    // get selected pointclouds
    std::vector<ccPointCloud*> clouds;
    clouds.reserve(m_selected.size());
    for (ccHObject* entity : m_selected) {
        // clouds only
        if (entity->isA(CV_TYPES::POINT_CLOUD)) {
            ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(entity);
            if (!cloud->hasNormals()) {
                return;
            }
            clouds.push_back(cloud);
        }
    }
    if (clouds.size() < 2) {
        return;
    }

    FastGlobalRegistrationDialog dialog(
            clouds, m_app ? m_app->getMainWindow() : nullptr);

    if (!dialog.exec()) {
        return;
    }

    // retrieve the reference clouds (= all the others)
    m_referenceCloud = dialog.getReferenceCloud();

    // retrieve the aligned clouds (= all the others)
    m_alignedClouds.reserve(clouds.size() - 1);
    for (ccPointCloud* cloud : clouds) {
        if (cloud != m_referenceCloud) m_alignedClouds.push_back(cloud);
    }

    // retrieve the feature radius
    m_featureRadius = dialog.getFeatureRadius();

    dialog.saveParameters();
}

int FastGlobalRegistrationFilter::compute() {
    if (m_alignedClouds.empty() || !m_referenceCloud || m_featureRadius <= 0) {
        assert(false);
        return -1;
    }
    if (!m_referenceCloud->hasNormals()) {
        assert(false);
        return -1;
    }

    // compute the feature vector for the reference cloud
    fgr::Features referenceFeatures;
    if (!ComputeFeatures(m_referenceCloud, referenceFeatures,
                         m_featureRadius)) {
        CVLog::Warning(
                "Failed to compute the reference cloud feature descriptors");
        return -53;
    }

    // convert the reference cloud to vectors of Eigen::Vector3f
    fgr::Points referencePoints;
    if (!ConverFromTo(*m_referenceCloud, referencePoints)) {
        return -1;
    }

    // now for each aligned cloud
    for (ccPointCloud* alignedCloud : m_alignedClouds) {
        if (!alignedCloud->hasNormals()) {
            assert(false);
            return -1;
        }

        // now compute the feature vector
        fgr::Features alignedFeatures;
        if (!ComputeFeatures(alignedCloud, alignedFeatures, m_featureRadius)) {
            CVLog::Warning("Failed to compute the point feature descriptors");
            return -1;
        }

        // now convert the cloud to vectors of Eigen::Vector3f
        fgr::Points alignedPoints;
        if (!ConverFromTo(*alignedCloud, alignedPoints)) {
            return -1;
        }

        ccGLMatrix ccTrans;
        try {
            fgr::CApp fgrProcess;
            fgrProcess.LoadFeature(referencePoints, referenceFeatures);
            fgrProcess.LoadFeature(alignedPoints, alignedFeatures);
            fgrProcess.NormalizePoints();
            fgrProcess.AdvancedMatching();
            if (!fgrProcess.OptimizePairwise(true)) {
                CVLog::Warning(
                        "Failed to perform pair-wise optimization (not enough "
                        "points)");
                return -1;
            }

            Eigen::Matrix4f trans = fgrProcess.GetOutputTrans();
            for (int i = 0; i < 16; ++i) {
                // both ccGLMatrix and Eigen::Matrix4f should use column-major
                // storage
                ccTrans.data()[i] = trans.data()[i];
            }
        } catch (...) {
            CVLog::Warning(
                    "Failed to determine the Global Registration matrix");
            return -1;
        }

        alignedCloud->setGLTransformation(ccTrans);

        CVLog::Print(
                tr("[Fast Global Registration] Resulting matrix for cloud %1")
                        .arg(alignedCloud->getName()));
        CVLog::Print(ccTrans.toString(12, ' '));  // full precision
        CVLog::Print(
                tr("Hint: copy it (CTRL+C) and apply it - or its inverse - on "
                   "any entity with the 'Edit > Apply transformation' tool"));
    }

    return 1;
}
