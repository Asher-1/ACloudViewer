// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "qPCLBase.h"

// CV_DB_LIB
#include <ecvPointCloud.h>

// PCL_PLUGIN_ALGORIGHM_LIB
#include <BasePclModule.h>

// FILTERS
#include <Filters/FastGlobalRegistrationFilter.h>
#include <Filters/GeneralFilters.h>
#include <Filters/MLSSmoothingUpsampling.h>
#include <Filters/ProjectionFilter.h>
#include <Filters/StatisticalOutliersRemover.h>

// FEATURES
#include <Features/ExtractSIFT.h>
#include <Features/NormalEstimation.h>

// SEGMENTATIONS
#include <Segmentations/DONSegmentation.h>
#include <Segmentations/EuclideanClusterSegmentation.h>
#include <Segmentations/MinimumCutSegmentation.h>
#include <Segmentations/RegionGrowingSegmentation.h>
#include <Segmentations/SACSegmentation.h>

// RECOGNITIONS
#include <Recognitions/CorrespondenceMatching.h>
#include <Recognitions/TemplateAlignment.h>

// SURFACES
#include <Surfaces/ConvexConcaveHullReconstruction.h>
#include <Surfaces/GreedyTriangulation.h>
#include <Surfaces/MarchingCubeReconstruction.h>
#include <Surfaces/PoissonReconstruction.h>
#if defined(WITH_PCL_NURBS)
#include <Surfaces/NurbsCurveFitting.h>
#include <Surfaces/NurbsSurfaceReconstruction.h>
#endif

qPCL::qPCL(QObject *parent /*=0*/)
    : QObject(parent), ccPclPluginInterface(":/toolbar/info.json") {}

void qPCL::stop() {
    while (!m_modules.empty()) {
        delete m_modules.back();
        m_modules.pop_back();
    }
}

void qPCL::handleNewEntity(ccHObject *entity) {
    assert(entity && m_app);
    m_app->addToDB(entity);
    m_app->zoomOnEntities(entity);
}

void qPCL::handleEntityChange(ccHObject *entity) {
    assert(entity && m_app);
    m_app->refreshSelected();
    m_app->updateUI();
}

void qPCL::handleErrorMessage(QString message) {
    if (m_app)
        m_app->dispToConsole(message, ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
}

QVector<QList<QAction *>> qPCL::getActions() {
    QVector<QList<QAction *>> allModuleActions;

    // ADD actions
    if (m_modules.empty() && m_moduleNames.empty()) {
        // ADD FILTERS
        QList<QAction *> filterActions;
        m_moduleNames.push_back(tr("Filters"));
        addPclModule(new GeneralFilters(), filterActions);
        addPclModule(new ProjectionFilter(), filterActions);
        addPclModule(new MLSSmoothingUpsampling(), filterActions);
        addPclModule(new StatisticalOutliersRemover(), filterActions);
        addPclModule(new FastGlobalRegistrationFilter(), filterActions);
        allModuleActions.push_back(filterActions);

        // ADD FEATURES
        QList<QAction *> featureActions;
        m_moduleNames.push_back(tr("Features"));
        addPclModule(new NormalEstimation(), featureActions);
        addPclModule(new ExtractSIFT(), featureActions);
        allModuleActions.push_back(featureActions);

        // ADD SURFACES
        QList<QAction *> surfaceActions;
        m_moduleNames.push_back(tr("Surfaces"));
        addPclModule(new MarchingCubeReconstruction(), surfaceActions);
        addPclModule(new GreedyTriangulation(), surfaceActions);
        addPclModule(new PoissonReconstruction(), surfaceActions);
#if defined(WITH_PCL_NURBS)
        addPclModule(new NurbsCurveFitting(), surfaceActions);
        addPclModule(new NurbsSurfaceReconstruction(), surfaceActions);
#endif
        addPclModule(new ConvexConcaveHullReconstruction(), surfaceActions);
        allModuleActions.push_back(surfaceActions);

        // ADD SEGMENTATION
        QList<QAction *> segmentActions;
        m_moduleNames.push_back(tr("Segmentations"));
        addPclModule(new DONSegmentation(), segmentActions);
        addPclModule(new MinimumCutSegmentation(), segmentActions);
        addPclModule(new RegionGrowingSegmentation(), segmentActions);
        addPclModule(new SACSegmentation(), segmentActions);
        addPclModule(new EuclideanClusterSegmentation(), segmentActions);
        allModuleActions.push_back(segmentActions);

        // ADD RECOGNITION
        QList<QAction *> recognitionActions;
        m_moduleNames.push_back(tr("Recognitions"));
        addPclModule(new TemplateAlignment(), recognitionActions);
        addPclModule(new CorrespondenceMatching(), recognitionActions);
        allModuleActions.push_back(recognitionActions);
    }

    return allModuleActions;
}

QVector<QString> qPCL::getModuleNames() { return m_moduleNames; }

int qPCL::addPclModule(BasePclModule *module, QList<QAction *> &actions) {
    assert(module);
    module->setMainAppInterface(m_app);

    QAction *action = module->getAction();
    if (!action) return 0;

    // module already inserted?
    if (std::find(m_modules.begin(), m_modules.end(), module) !=
        m_modules.end())
        return 0;

    actions.push_back(action);
    m_modules.push_back(module);

    // connect signals
    connect(module, &BasePclModule::newEntity, this, &qPCL::handleNewEntity);
    connect(module, &BasePclModule::entityHasChanged, this,
            &qPCL::handleEntityChange);
    connect(module, &BasePclModule::newErrorMessage, this,
            &qPCL::handleErrorMessage);

    return 1;
}

void qPCL::onNewSelection(const ccHObject::Container &selectedEntities) {
    for (size_t i = 0; i < m_modules.size(); ++i)
        m_modules[i]->updateSelectedEntities(selectedEntities);

    for (size_t i = 0; i < m_modules.size(); ++i)
        m_modules[i]->updateSelectedEntities(selectedEntities);
}
