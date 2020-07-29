//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER PLUGIN: qPCL                         #
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
//#                        COPYRIGHT: DAHAI LU                             #
//#                                                                        #
//##########################################################################
//
#include "qPCL.h"

// ECV_DB_LIB
#include <ecvPointCloud.h>

// PCL_PLUGIN_ALGORIGHM_LIB
#include <BasePclModule.h>

// FILTERS
#include <Filters/GeneralFilters.h>
#include <Filters/ProjectionFilter.h>
#include <Filters/MLSSmoothingUpsampling.h>
#include <Filters/StatisticalOutliersRemover.h>

// FEATURES
#include <Features/ExtractSIFT.h>
#include <Features/NormalEstimation.h>

// SEGMENTATIONS
#include <Segmentations/DONSegmentation.h>
#include <Segmentations/MinimumCutSegmentation.h>
#include <Segmentations/RegionGrowingSegmentation.h>
#include <Segmentations/SACSegmentation.h>
#include <Segmentations/EuclideanClusterSegmentation.h>

// RECOGNITIONS
#include <Recognitions/TemplateAlignment.h>
#include <Recognitions/CorrespondenceMatching.h>

// SURFACES
#include <Surfaces/MarchingCubeReconstruction.h>
#include <Surfaces/GreedyTriangulation.h>
#include <Surfaces/PoissonReconstruction.h>
#include <Surfaces/NurbsCurveFitting.h>
#include <Surfaces/NurbsSurfaceReconstruction.h>
#include <Surfaces/ConvexConcaveHullReconstruction.h>

qPCL::qPCL(QObject* parent/*=0*/)
	: QObject(parent)
	, ccPclPluginInterface(":/toolbar/info.json")
{
}

void qPCL::stop()
{
	while (!m_modules.empty())
	{
		delete m_modules.back();
		m_modules.pop_back();
	}
}

void qPCL::handleNewEntity(ccHObject* entity)
{
	assert(entity && m_app);
	m_app->addToDB(entity);
	m_app->zoomOnEntities(entity);
}

void qPCL::handleEntityChange(ccHObject* entity)
{
	assert(entity && m_app);
	m_app->refreshSelected();
	m_app->updateUI();
}

void qPCL::handleErrorMessage(QString message)
{
	if (m_app)
		m_app->dispToConsole(message, ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
}

QVector<QList<QAction *>> qPCL::getActions()
{
	QVector<QList<QAction *>> allModuleActions;

	// ADD actions
	if (m_modules.empty() && m_moduleNames.empty())
	{
		// ADD FILTERS
		QList<QAction *> filterActions;
		m_moduleNames.push_back(tr("Filters"));
		addPclModule(new GeneralFilters(), filterActions);
		addPclModule(new ProjectionFilter(), filterActions);
		addPclModule(new MLSSmoothingUpsampling(), filterActions);
		addPclModule(new StatisticalOutliersRemover(), filterActions);
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
		addPclModule(new NurbsCurveFitting(), surfaceActions);
		addPclModule(new NurbsSurfaceReconstruction(), surfaceActions);
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

QVector<QString> qPCL::getModuleNames()
{
	return m_moduleNames;
}

int qPCL::addPclModule(BasePclModule* module, QList<QAction *> &actions)
{
	assert(module);
	module->setMainAppInterface(m_app);

	QAction* action = module->getAction();
	if (!action)
		return 0;

	// module already inserted?
	if (std::find(m_modules.begin(), m_modules.end(), module) != m_modules.end())
		return 0;

	actions.push_back(action);
	m_modules.push_back(module);

	//connect signals
	connect(module, SIGNAL(newEntity(ccHObject*)),			this,	SLOT(handleNewEntity(ccHObject*)));
	connect(module, SIGNAL(entityHasChanged(ccHObject*)),	this,	SLOT(handleEntityChange(ccHObject*)));
	connect(module, SIGNAL(newErrorMessage(QString)),		this,	SLOT(handleErrorMessage(QString)));

	return 1;
}

void qPCL::onNewSelection(const ccHObject::Container& selectedEntities)
{
	for (size_t i=0; i<m_modules.size(); ++i)
		m_modules[i]->updateSelectedEntities(selectedEntities);	
	
	for (size_t i=0; i< m_modules.size(); ++i)
		m_modules[i]->updateSelectedEntities(selectedEntities);
}
