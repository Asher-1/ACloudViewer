// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TemplateAlignment.h"

#include <Utils/cc2sm.h>
#include <Utils/sm2cc.h>

#include "PclUtils/PCLModules.h"
#include "dialogs/TemplateAlignmentDialog.h"

// CV_DB_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>

TemplateAlignment::TemplateAlignment()
    : BasePclModule(PclModuleDescription(
              tr("Template Alignment"),
              tr("Template Alignment"),
              tr("Template Alignment from clouds"),
              ":/toolbar/PclAlgorithms/icons/templateAlignment.png")),
      m_dialog(nullptr),
      m_templateMatch(nullptr),
      m_targetCloud(nullptr),
      m_useVoxelGrid(false),
      m_leafSize(0.005f),
      m_normalRadius(0.02f),
      m_featureRadius(0.02f),
      m_minSampleDistance(0.05f),
      m_maxCorrespondenceDistance(0.01f * 0.01f),
      m_maxIterations(500),
      m_maxThreadCount(1) {}

TemplateAlignment::~TemplateAlignment() {
    // we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr) delete m_dialog;
    if (m_templateMatch) {
        delete m_templateMatch;
    }
}

int TemplateAlignment::checkSelected() {
    // do we have a selected cloud?
    int have_cloud = isFirstSelectedCcPointCloud();
    if (have_cloud != 1) return -11;

    return 1;
}

int TemplateAlignment::openInputDialog() {
    // initialize the dialog object
    if (!m_dialog) m_dialog = new TemplateAlignmentDialog(m_app);

    m_dialog->refreshCloudComboBox();
    if (!m_dialog->exec()) return 0;

    m_dialog->saveParamsToPersistentSettings();

    return 1;
}

void TemplateAlignment::getParametersFromDialog() {
    if (!m_dialog) return;

    // get PCLModules::FeatureCloud parameters
    m_normalRadius = m_dialog->getNormalRadius();
    m_featureRadius = m_dialog->getFeatureRadius();
    m_maxThreadCount = m_dialog->getMaxIterations();
    // get PCLModules::TemplateMatching parameters
    m_maxIterations = m_dialog->getMaxIterations();
    m_minSampleDistance = m_dialog->getMinSampleDistance();
    m_maxCorrespondenceDistance = m_dialog->getMaxCorrespondenceDistance();

    if (!m_templateMatch) {
        m_templateMatch = new PCLModules::TemplateMatching();
    }
    m_templateMatch->clear();

    // PCLModules::TemplateMatching parameters
    m_templateMatch->setmaxIterations(m_maxIterations);
    m_templateMatch->setminSampleDis(m_minSampleDistance);
    m_templateMatch->setmaxCorrespondenceDis(m_maxCorrespondenceDistance);

    // get the parameters from the dialog
    m_leafSize = m_dialog->getVoxelGridLeafSize();
    m_useVoxelGrid = m_leafSize > 0 ? true : false;

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

    ccPointCloud* cloud1 = m_dialog->getTemplate1Cloud();
    ccPointCloud* cloud2 = m_dialog->getTemplate2Cloud();
    m_templateClouds.clear();
    {
        if (!cloud1 && !cloud2) {
            if (m_app)
                m_app->dispToConsole(tr("At least one cloud (class #1 or #2) "
                                        "was not defined!"),
                                     ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
            return;
        }
        assert(cloud1 != cloud2);
        if (cloud1) m_templateClouds.push_back(cloud1);
        if (cloud2) m_templateClouds.push_back(cloud2);
    }

    m_targetCloud = m_dialog->getEvaluationCloud();
}

int TemplateAlignment::checkParameters() {
    if (!m_targetCloud || m_templateClouds.size() == 0) {
        return -52;
    }

    if (m_featureRadius < m_normalRadius) {
        return -52;
    }
    return 1;
}

int TemplateAlignment::compute() {
    // 1. Load the object templates
    for (size_t i = 0; i < m_templateClouds.size(); ++i) {
        PointCloudT::Ptr cloudFiltered =
                cc2smReader(m_templateClouds[i]).getXYZ2();
        if (cloudFiltered->width * cloudFiltered->height != 0) {
            // Assign to the template FeatureCloud
            PCLModules::FeatureCloud template_cloud;

            template_cloud.setmaxThreadCount(m_maxThreadCount);
            template_cloud.setNormalRadius(m_normalRadius);
            template_cloud.setFeatureRadius(m_featureRadius);
            template_cloud.setInputCloud(cloudFiltered);
            m_templateMatch->addTemplateCloud(template_cloud);
        }
    }

    // 2. Load the target cloud
    {
        PointCloudT::Ptr cloudFiltered = cc2smReader(m_targetCloud).getXYZ2();
        // down sampling the template point cloud 1
        if (m_useVoxelGrid) {
            PointCloudT::Ptr tempCloud(new PointCloudT);
            if (!PCLModules::VoxelGridFilter<PointT>(cloudFiltered, tempCloud,
                                                     m_leafSize, m_leafSize,
                                                     m_leafSize)) {
                return -1;
            }
            cloudFiltered = tempCloud;
        }

        if (!cloudFiltered) {
            return -1;
        }

        // Assign to the target FeatureCloud
        PCLModules::FeatureCloud target_cloud;
        target_cloud.setmaxThreadCount(m_maxThreadCount);
        target_cloud.setNormalRadius(m_normalRadius);
        target_cloud.setFeatureRadius(m_featureRadius);

        target_cloud.setInputCloud(cloudFiltered);
        m_templateMatch->setTargetCloud(target_cloud);
    }

    // 3. Find the best template alignment
    PCLModules::TemplateMatching::Result best_alignment;
    int best_index = m_templateMatch->findBestAlignment(best_alignment);
    const PCLModules::FeatureCloud* best_template =
            m_templateMatch->getTemplateCloud(best_index);
    if (!best_template) {
        return -1;
    }

    // Print the alignment fitness score (values less than 0.00002 are good)
    CVLog::Print(
            tr("(values less than 0.00002 are good) Best fitness score: %1")
                    .arg(best_alignment.fitness_score));

    ccGLMatrixd transMat(best_alignment.final_transformation.data());

    // Save the aligned template for visualization
    PCLCloud out_cloud_sm;
    TO_PCL_CLOUD(*best_template->getPointCloud(), out_cloud_sm);
    if (out_cloud_sm.height * out_cloud_sm.width == 0) {
        // cloud is empty
        return -53;
    }

    ccPointCloud* out_cloud_cc = pcl2cc::Convert(out_cloud_sm);
    if (!out_cloud_cc) {
        // conversion failed (not enough memory?)
        return -1;
    }

    // apply transformation and colors
    applyTransformation(out_cloud_cc, transMat);
    {
        out_cloud_cc->setRGBColor(ecvColor::darkBlue);
        out_cloud_cc->showColors(true);
        out_cloud_cc->showSF(false);
    }

    // copy global shift & scale and set name
    ccPointCloud* cloud = nullptr;
    {
        if (best_index == 0) {
            cloud = m_dialog->getTemplate1Cloud();
        } else if (best_index == 1) {
            cloud = m_dialog->getTemplate2Cloud();
        }

        if (cloud) {
            QString outName = cloud->getName() + "-alignment";
            out_cloud_cc->setName(outName);
            // copy global shift & scale
            out_cloud_cc->setGlobalScale(cloud->getGlobalScale());
            out_cloud_cc->setGlobalShift(cloud->getGlobalShift());

            if (cloud->getParent()) cloud->getParent()->addChild(out_cloud_cc);
        }
    }

    emit newEntity(out_cloud_cc);

    return 1;
}

void TemplateAlignment::applyTransformation(ccHObject* entity,
                                            const ccGLMatrixd& mat) {
    entity->setGLTransformation(ccGLMatrix(mat.data()));
    // DGM FIXME: we only test the entity own bounding box (and we update its
    // shift & scale info) but we apply the transformation to all its children?!
    entity->applyGLTransformation_recursive();
    CVLog::Print(tr("[ApplyTransformation] Applied transformation matrix:"));
    CVLog::Print(mat.toString(12, ' '));  // full precision
    CVLog::Print(
            tr("Hint: copy it (CTRL+C) and apply it - or its inverse - on any "
               "entity with the 'Edit > Apply transformation' tool"));
}

QString TemplateAlignment::getErrorMessage(int errorCode) {
    switch (errorCode) {
            // THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS
            // CODES ALSO IN DERIVED CLASSES DEFULAT MUST BE ""

        case -51:
            return tr(
                    "Selected entity does not have any suitable scalar field "
                    "or RGB.");
        case -52:
            return tr(
                    "Wrong Parameters. One or more parameters cannot be "
                    "accepted");
        case -53:
            return tr(
                    "Template Alignment does not returned any point. Try "
                    "relaxing your parameters");
    }

    return BasePclModule::getErrorMessage(errorCode);
}
