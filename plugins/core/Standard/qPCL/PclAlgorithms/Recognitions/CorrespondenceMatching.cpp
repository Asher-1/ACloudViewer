// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CorrespondenceMatching.h"

#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "dialogs/CorrespondenceMatchingDialog.h"

// CV_DB_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>
#include <sstream>

CorrespondenceMatching::CorrespondenceMatching()
    : BasePclModule(PclModuleDescription(
              tr("Correspondence Matching"),
              tr("Correspondence Matching"),
              tr("Correspondence Matching from clouds"),
              ":/toolbar/PclAlgorithms/icons/correspondence_grouping.png")),
      m_dialog(nullptr),
      m_sceneCloud(nullptr),
      m_useVoxelGrid(false),
      m_gcMode(true),
      m_verification(true),
      m_maxThreadCount(1),
      m_leafSize(0.005f),
      m_modelSearchRadius(0.02f),
      m_sceneSearchRadius(0.03f),
      m_shotDescriptorRadius(0.03f),
      m_normalKSearch(10.0f),
      m_consensusResolution(0.01f),
      m_gcMinClusterSize(20.0f),
      m_lrfRadius(0.015f),
      m_houghBinSize(0.01f),
      m_houghThreshold(5.0f) {}

CorrespondenceMatching::~CorrespondenceMatching() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
}

int CorrespondenceMatching::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int CorrespondenceMatching::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog) m_dialog = new CorrespondenceMatchingDialog(m_app);
    m_dialog->refreshCloudComboBox();

    if (!m_dialog->exec()) return 0;

    m_dialog->saveParamsToPersistentSettings();

    return 1;
}

void CorrespondenceMatching::getParametersFromDialog() {
    if (!m_dialog) return;

    // get the parameters from the dialog
    m_gcMode = m_dialog->isGCActivated();
    m_leafSize = m_dialog->getVoxelGridLeafSize();
    m_useVoxelGrid = m_leafSize > 0 ? true : false;
    m_maxThreadCount = m_dialog->getMaxThreadCount();

    m_verification = m_dialog->getVerificationFlag();

    m_modelSearchRadius = m_dialog->getModelSearchRadius();
    m_sceneSearchRadius = m_dialog->getSceneSearchRadius();
    m_shotDescriptorRadius = m_dialog->getShotDescriptorRadius();
    m_normalKSearch = m_dialog->getNormalKSearch();
    m_consensusResolution = m_dialog->getGcConsensusSetResolution();
    m_gcMinClusterSize = m_dialog->getGcMinClusterSize();
    m_lrfRadius = m_dialog->getHoughLRFRadius();
    m_houghBinSize = m_dialog->getHoughBinSize();
    m_houghThreshold = m_dialog->getHoughThreshold();

    // get scales
    m_scales.clear();
    {
        if (!m_dialog->getScales(m_scales)) {
            m_app->dispToConsole(tr("Invalid scale parameters!"),
                                 ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
        // make sure values are in descending order!
        std::sort(m_scales.begin(), m_scales.end(), std::greater<float>());
    }

    ccPointCloud* cloud1 = m_dialog->getModelCloudByIndex(1);
    ccPointCloud* cloud2 = m_dialog->getModelCloudByIndex(2);
    m_modelClouds.clear();
    {
        if (!cloud1 && !cloud2) {
            if (m_app)
                m_app->dispToConsole(tr("At least one cloud (model #1 or #2) "
                                        "was not defined!"),
                                     ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
        assert(cloud1 != cloud2);
        if (cloud1) m_modelClouds.push_back(cloud1);
        if (cloud2) m_modelClouds.push_back(cloud2);
    }

    m_sceneCloud = m_dialog->getEvaluationCloud();
}

int CorrespondenceMatching::checkParameters() {
    if (!m_sceneCloud || m_modelClouds.size() == 0) {
        return -52;
    }
    return 1;
}

int CorrespondenceMatching::compute() {
    typedef PointT PointType;
    typedef NormalT NormalType;
    typedef pcl::ReferenceFrame RFType;
    typedef pcl::SHOT352 DescriptorType;

    std::vector<std::vector<ccGLMatrixd>> transMatVecs;
    std::vector<std::vector<Eigen::Matrix4f,
                            Eigen::aligned_allocator<Eigen::Matrix4f>>>
            rotoTranslationsVec;
    std::vector<std::vector<pcl::Correspondences>> clusteredCorrs;
    std::vector<std::vector<pcl::PointCloud<PointType>::ConstPtr>> instancesVec;
    std::vector<std::vector<bool>> hypothesesMaskVec;

    PointCloudT::Ptr sceneKeypoints(new PointCloudT());
    pcl::PointCloud<NormalType>::Ptr sceneNormals(
            new pcl::PointCloud<NormalType>());
    pcl::PointCloud<DescriptorType>::Ptr sceneDescriptors(
            new pcl::PointCloud<DescriptorType>());
    std::vector<pcl::CorrespondencesPtr> modelSceneCorrs;
    std::vector<pcl::PointCloud<DescriptorType>::Ptr> modelsDescriptors;
    std::vector<pcl::PointCloud<NormalType>::Ptr> modelsWithNormal;
    std::vector<PointCloudT::Ptr> modelsKeypoints;
    std::vector<PointCloudT::Ptr> modelClouds;

    // 1. Load the scene cloud
    PointCloudT::Ptr scene = cc2smReader(m_sceneCloud).getXYZ2();
    {
        // down sampling the model point cloud 1
        if (m_useVoxelGrid) {
            PointCloudT::Ptr tempCloud(new PointCloudT);
            if (!PCLModules::VoxelGridFilter<PointType>(
                        scene, tempCloud, m_leafSize, m_leafSize, m_leafSize)) {
                return -1;
            }
            scene = tempCloud;
        }

        if (!scene) {
            return -1;
        }
    }

    // 2. Compute the model and scene normals
    for (size_t i = 0; i < m_modelClouds.size(); ++i) {
        PointCloudT::Ptr modelCloud = cc2smReader(m_modelClouds[i]).getXYZ2();
        if (modelCloud->width * modelCloud->height != 0) {
            modelClouds.push_back(modelCloud);
            pcl::PointCloud<NormalType>::Ptr model_normals(
                    new pcl::PointCloud<NormalType>());
            if (PCLModules::ComputeNormals<PointType, NormalType>(
                        modelCloud, model_normals, m_normalKSearch, true,
                        m_maxThreadCount)) {
                modelsWithNormal.push_back(model_normals);
            }
        }
    }
    if (!PCLModules::ComputeNormals<PointType, NormalType>(
                scene, sceneNormals, m_normalKSearch, true, m_maxThreadCount)) {
        return -1;
    }

    // 3. Downsample Clouds to Extract keypoints
    if (!PCLModules::GetUniformSampling<PointType>(scene, sceneKeypoints,
                                                   m_sceneSearchRadius)) {
        return -1;
    }
    CVLog::Print(tr("Scene total points: %1; Selected Keypoints: %2")
                         .arg(scene->size())
                         .arg(sceneKeypoints->size()));
    for (size_t i = 0; i < modelClouds.size(); ++i) {
        PointCloudT::Ptr model_keypoints(new PointCloudT());
        if (!PCLModules::GetUniformSampling<PointType>(
                    modelClouds[i], model_keypoints, m_modelSearchRadius)) {
            return -1;
        }
        modelsKeypoints.push_back(model_keypoints);
        CVLog::Print(tr("Model %1 total points: %2; Selected Keypoints: %3")
                             .arg(i + 1)
                             .arg(modelClouds[i]->size())
                             .arg(model_keypoints->size()));
    }

    // 4. Compute Descriptor for keypoints
    if (!PCLModules::EstimateShot<PointType, NormalType, DescriptorType>(
                scene, sceneKeypoints, sceneNormals, sceneDescriptors,
                m_shotDescriptorRadius, m_maxThreadCount)) {
        return -1;
    }
    for (size_t i = 0; i < modelClouds.size(); ++i) {
        pcl::PointCloud<DescriptorType>::Ptr model_descriptors(
                new pcl::PointCloud<DescriptorType>());
        if (!PCLModules::EstimateShot<PointType, NormalType, DescriptorType>(
                    modelClouds[i], modelsKeypoints[i], modelsWithNormal[i],
                    model_descriptors, m_shotDescriptorRadius,
                    m_maxThreadCount)) {
            return -1;
        }
        modelsDescriptors.push_back(model_descriptors);
    }

    // 5. Find Model-Scene Correspondences with KdTree
    pcl::KdTreeFLANN<DescriptorType> match_search;
    for (size_t j = 0; j < modelClouds.size(); ++j) {
        pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
        match_search.setInputCloud(modelsDescriptors[j]);

        //  For each scene keypoint descriptor, find nearest neighbor into the
        //  model keypoints descriptor cloud and add it to the correspondences
        //  vector.
        for (size_t i = 0; i < sceneDescriptors->size(); ++i) {
            std::vector<int> neigh_indices(1);
            std::vector<float> neigh_sqr_dists(1);
            if (!std::isfinite(sceneDescriptors->at(i)
                                       .descriptor[0]))  // skipping NaNs
            {
                continue;
            }
            int found_neighs = match_search.nearestKSearch(
                    sceneDescriptors->at(i), 1, neigh_indices, neigh_sqr_dists);
            if (found_neighs == 1 &&
                neigh_sqr_dists[0] <
                        0.25f)  //  add match only if the squared descriptor
                                //  distance is less than 0.25 (SHOT descriptor
                                //  distances are between 0 and 1 by design)
            {
                pcl::Correspondence corr(neigh_indices[0], static_cast<int>(i),
                                         neigh_sqr_dists[0]);
                model_scene_corrs->push_back(corr);
            }
        }
        CVLog::Print(tr("Correspondences found for model %1 : %2")
                             .arg(j + 1)
                             .arg(model_scene_corrs->size()));
        modelSceneCorrs.push_back(model_scene_corrs);
    }

    // 6. Actual Clustering
    for (size_t j = 0; j < modelClouds.size(); ++j) {
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
                transformMatrix;
        std::vector<pcl::Correspondences> clustered_corrs;

        pcl::CorrespondencesPtr model_scene_corrs = modelSceneCorrs[j];
        PointCloudT::Ptr model_keypoints = modelsKeypoints[j];
        PointCloudT::Ptr model = modelClouds[j];
        pcl::PointCloud<NormalType>::Ptr model_normals = modelsWithNormal[j];

        if (m_gcMode) {
            if (!PCLModules::EstimateGeometricConsistencyGrouping<PointType,
                                                                  PointType>(
                        model_keypoints, sceneKeypoints, model_scene_corrs,
                        transformMatrix, clustered_corrs, m_consensusResolution,
                        m_gcMinClusterSize)) {
                return -1;
            }
        } else {
            //
            //  Compute (Keypoints) Reference Frames only for Hough
            //
            pcl::PointCloud<RFType>::Ptr model_rf(
                    new pcl::PointCloud<RFType>());
            pcl::PointCloud<RFType>::Ptr scene_rf(
                    new pcl::PointCloud<RFType>());

            if (!PCLModules::EstimateLocalReferenceFrame<PointType, NormalType,
                                                         RFType>(
                        model, model_keypoints, model_normals, model_rf,
                        m_lrfRadius)) {
                return -1;
            }

            if (!PCLModules::EstimateLocalReferenceFrame<PointType, NormalType,
                                                         RFType>(
                        scene, sceneKeypoints, sceneNormals, scene_rf,
                        m_lrfRadius)) {
                return -1;
            }

            //  Clustering
            if (!PCLModules::EstimateHough3DGrouping<PointType, PointType,
                                                     RFType, RFType>(
                        model_keypoints, sceneKeypoints, model_rf, scene_rf,
                        model_scene_corrs, transformMatrix, clustered_corrs,
                        m_houghBinSize, m_houghThreshold)) {
                return -1;
            }
        }

        rotoTranslationsVec.push_back(transformMatrix);
        clusteredCorrs.push_back(clustered_corrs);
    }

    // 7. verify result with icp and global
    for (size_t j = 0; j < rotoTranslationsVec.size(); ++j) {
        if (rotoTranslationsVec[j].size() <= 0) {
            CVLog::Print(tr("No instances found in Model %1 ").arg(j + 1));
            continue;
        }

        CVLog::Print(tr("Model %1 instances found number : %2")
                             .arg(j + 1)
                             .arg(rotoTranslationsVec[j].size()));

        /**
         * Generates clouds for each instances found
         */
        std::vector<pcl::PointCloud<PointType>::ConstPtr> instances;

        for (size_t i = 0; i < rotoTranslationsVec[j].size(); ++i) {
            pcl::PointCloud<PointType>::Ptr rotated_model(
                    new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*modelClouds[j], *rotated_model,
                                     rotoTranslationsVec[j][i]);
            instances.push_back(rotated_model);
        }
        instancesVec.push_back(instances);

        std::vector<bool>
                hypotheses_mask;  // Mask Vector to identify positive hypotheses
        if (m_verification) {
            /**
             * ICP
             */
            std::vector<pcl::PointCloud<PointType>::ConstPtr>
                    registered_instances;
            {
                CVLog::Print(tr("--- ICP Start ---------"));

                for (size_t i = 0; i < rotoTranslationsVec[j].size(); ++i) {
                    pcl::PointCloud<PointType>::Ptr registered(
                            new pcl::PointCloud<PointType>);
                    if (PCLModules::ICPRegistration<PointType, PointType>(
                                scene, instances[i], registered,
                                m_icpMaxIterations, m_icpCorrDistance)) {
                        CVLog::Print(tr("Model %1 Instances %2 Aligned!")
                                             .arg(j + 1)
                                             .arg(i + 1));
                    } else {
                        CVLog::Print(tr("Model %1 Instances %2 Not Aligned!")
                                             .arg(j + 1)
                                             .arg(i + 1));
                    }
                    registered_instances.push_back(registered);
                }
                CVLog::Print(tr("--- ICP End ---------"));
            }

            /**
             * Hypothesis Verification
             */
            CVLog::Print(
                    tr("--- Hypotheses bugs and do not support! ---------"));
            //			CVLog::Print(tr("--- Hypotheses Verification
            // Start
            //---------")); 			if
            //(!PCLModules::GetHypothesesVerification<PointType, PointType>(
            //				scene, registered_instances,
            // hypotheses_mask, clusterReg, inlierThreshold,
            // occlusionThreshold, radiusClutter, regularizer, radiusNormals,
            // detectClutter))
            //			{
            //				return -1;
            //			}
            //			for (int i = 0; i < hypotheses_mask.size(); i++)
            //			{
            //				if (hypotheses_mask[i])
            //				{
            //					CVLog::Print(tr("Model %1
            // Instances %2 is GOOD!").arg(j
            //+ 1).arg(i + 1));
            //				}
            //				else
            //				{
            //					CVLog::Print(tr("Model %1
            // Instances %2 is bad and will be discarded!").arg(j + 1).arg(i +
            // 1));
            //				}
            //			}
            //			CVLog::Print(tr("--- Hypotheses Verification End
            //---------"));
        } else {
            for (size_t i = 0; i < rotoTranslationsVec[j].size(); ++i) {
                hypotheses_mask.push_back(true);
            }
        }

        hypothesesMaskVec.push_back(hypotheses_mask);
    }

    // 8. export result
    ccHObject* ecvGroup =
            new ccHObject(tr("correspondence-grouping-cluster(s)"));
    ecvGroup->setVisible(true);

    for (int j = 0; j < static_cast<int>(instancesVec.size()); ++j) {
        for (int i = 0; i < static_cast<int>(instancesVec[j].size()); ++i) {
            // Save the aligned model for visualization
            if (hypothesesMaskVec[j][i]) {
                PCLCloud out_cloud_sm;
                TO_PCL_CLOUD(*instancesVec[j][i], out_cloud_sm);
                if (out_cloud_sm.height * out_cloud_sm.width == 0) {
                    // cloud is empty
                    return -53;
                }

                ccPointCloud* out_cloud_cc = pcl2cc::Convert(out_cloud_sm);
                if (!out_cloud_cc) {
                    // conversion failed (not enough memory?)
                    return -1;
                }

                // copy global shift & scale and set name
                ccPointCloud* cloud = m_dialog->getModelCloudByIndex(j + 1);
                ecvColor::Rgb col = ecvColor::Generator::Random();
                {
                    out_cloud_cc->setRGBColor(col);
                    out_cloud_cc->showColors(true);
                    out_cloud_cc->showSF(false);

                    if (cloud) {
                        QString outName = tr("%1-correspondence-%2-%3")
                                                  .arg(cloud->getName())
                                                  .arg(j + 1)
                                                  .arg(i + 1);
                        out_cloud_cc->setName(outName);
                        // copy global shift & scale
                        out_cloud_cc->setGlobalScale(cloud->getGlobalScale());
                        out_cloud_cc->setGlobalShift(cloud->getGlobalShift());
                    }
                }
                if (out_cloud_cc) {
                    ecvGroup->addChild(out_cloud_cc);
                }
            }
        }
    }

    if (ecvGroup->getChildrenNumber() == 0) {
        delete ecvGroup;
        ecvGroup = nullptr;
        return -51;
    } else {
        if (m_sceneCloud->getParent())
            m_sceneCloud->getParent()->addChild(ecvGroup);

        emit newEntity(ecvGroup);
    }

    return 1;
}

void CorrespondenceMatching::applyTransformation(ccHObject* entity,
                                                 const ccGLMatrixd& mat) {
    entity->setGLTransformation(ccGLMatrix(mat.data()));
    // DGM FIXME: we only test the entity own bounding box (and we update its
    // shift & scale info) but we apply the transformation to all its children?!
    entity->applyGLTransformation_recursive();
}

QString CorrespondenceMatching::getErrorMessage(int errorCode) {
    switch (errorCode) {
            // THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS
            // CODES ALSO IN DERIVED CLASSES DEFULAT MUST BE ""

        case -51:
            return tr("Cannot match anything by this model !");
        case -52:
            return tr(
                    "Wrong Parameters. One or more parameters cannot be "
                    "accepted");
        case -53:
            return tr(
                    "Correspondence Matching does not returned any point. Try "
                    "relaxing your parameters");
    }

    return BasePclModule::getErrorMessage(errorCode);
}
