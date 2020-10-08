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
//#                         COPYRIGHT: Asher                               #
//#                                                                        #
//##########################################################################
//

// LOCAL
#include "SACSegmentation.h"
#include "dialogs/SACSegmentationDlg.h"
#include "PclUtils/PCLCloud.h"
#include "PclUtils/PCLModules.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"
#include "Tools/ecvTools.h"  // must below above three

// ECV_DB_LIB
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>
#include <sstream>

int extractRecursive(
	PointCloudT::Ptr xyzCloud,
	PointCloudT::Ptr cloudRemained,
	std::vector<PointCloudT::Ptr> &cloudExtractions,
	int maxIterations,
	float probability,
	float minRadiusLimits,
	float maxRadiusLimits,
	float distanceThreshold,
	int methodType,
	int modelType,
	float normalDisWeight,
	float maxRemainingRatio,
	bool exportExtraction,
	bool recursive = false,
	ecvMainAppInterface* app = nullptr);

SACSegmentation::SACSegmentation()
	: BasePclModule(PclModuleDescription(tr("SAC Segmentation"),
										 tr("SAC Segmentation"),
										 tr("SAC Segmentation from clouds"),
										 ":/toolbar/PclAlgorithms/icons/SAC_Segmentation.png"))
	, m_dialog(0)
	, m_distanceThreshold(0.01f)
	, m_normalDisWeight(0.1f)
	, m_probability(0.95f)
	, m_minRadiusLimits(0.01f)
	, m_maxRadiusLimits(10.0f)
	, m_maxIterations(100)
	, m_methodType(0)
	, m_modelType(0)
	, m_exportExtraction(true)
	, m_exportRemaining(true)
	, m_maxRemainingRatio(0.3f)
	, m_recursiveMode(false)
	, m_useVoxelGrid(false)
	, m_leafSize(0.01f)
{
}

SACSegmentation::~SACSegmentation()
{
	//we must delete parent-less dialogs ourselves!
	if (m_dialog && m_dialog->parent() == 0)
		delete m_dialog;
}

int SACSegmentation::checkSelected()
{
	//do we have a selected cloud?
	int have_cloud = isFirstSelectedCcPointCloud();
	if (have_cloud != 1)
		return -11;

	return 1;
}

int SACSegmentation::openInputDialog()
{
	//initialize the dialog object
	if (!m_dialog)
		m_dialog = new SACSegmentationDlg(m_app ? m_app->getActiveWindow() : 0);

	if (!m_dialog->exec())
		return 0;

	return 1;
}

void SACSegmentation::getParametersFromDialog()
{
	if (!m_dialog)
		return;

	//get the parameters from the dialog
	m_modelType = m_dialog->modelTypeCombo->currentIndex();
	m_methodType = m_dialog->methodTypeCombo->currentIndex();

	m_distanceThreshold = static_cast<float>(m_dialog->disThresholdSpinBox->value());
	m_probability = static_cast<float>(m_dialog->probabilitySpinBox->value());
	m_minRadiusLimits = static_cast<float>(m_dialog->minRadiusSpinBox->value());
	m_maxRadiusLimits = static_cast<float>(m_dialog->maxRadiusSpinBox->value());
	m_maxIterations = m_dialog->maxIterationspinBox->value();
	m_normalDisWeight = static_cast<float>(m_dialog->normalDisWeightSpinBox->value());

	m_useVoxelGrid = m_dialog->useVoxelGridCheckBox->isChecked();
	m_leafSize = static_cast<float>(m_dialog->leafSizeSpinBox->value());
	m_recursiveMode = m_dialog->recursiveModeCheckBox->isChecked();
	m_maxRemainingRatio = static_cast<float>(m_dialog->maxRemainingRatioSpinBox->value());
	m_exportExtraction = m_dialog->exportExtractionCheckBox->isChecked();
	m_exportRemaining = m_dialog->exportRemainingCheckBox->isChecked();
}

int SACSegmentation::checkParameters()
{
	if (m_minRadiusLimits > m_maxRadiusLimits)
	{
		return -52;
	}

	if (!m_exportExtraction && !m_exportRemaining)
	{
		return -51;
	}

	return 1;
}

int SACSegmentation::compute()
{
	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (!cloud)
		return -1;

	// get xyz as pcl point cloud
	PointCloudT::Ptr cloudFiltered = cc2smReader(cloud).getXYZ2();
	if (!cloudFiltered)
		return -1;

	// voxel grid filter
	if (m_useVoxelGrid)
	{
		PointCloudT::Ptr tempCloud(new PointCloudT);
		if (!PCLModules::VoxelGridFilter<PointT>(cloudFiltered, tempCloud,
			m_leafSize, m_leafSize, m_leafSize))
		{
			return -1;
		}
		cloudFiltered = tempCloud;
	}

	PointCloudT::Ptr cloudRemained(new PointCloudT());
	std::vector<PointCloudT::Ptr> cloudExtractions;

	if (!extractRecursive(cloudFiltered, cloudRemained, cloudExtractions, 
		m_maxIterations, m_probability, m_minRadiusLimits, m_maxRadiusLimits, m_distanceThreshold,
		m_methodType, m_modelType, m_normalDisWeight, m_maxRemainingRatio, m_exportExtraction,
		m_recursiveMode, m_app))
	{
		return -53;
	}

	bool error = false;
	ccHObject* group = ecvTools::GetSegmentationGroup(cloud,
						m_exportRemaining ? cloudRemained : nullptr,
						cloudExtractions, true, error);

	if (group)
	{
		if (m_recursiveMode)
		{
			group->setName(group->getName() +
				tr("-MaxRemainingRatio[%1]").arg(m_maxRemainingRatio));
		}
		else
		{
			group->setName(group->getName() +
				tr("-SingleSegmentation[Distance Threshold %1]").arg(m_distanceThreshold));
		}

		unsigned count = group->getChildrenNumber();
		m_app->dispToConsole(
			tr("[SACSegmentation::compute] %1 extracted segment(s) where created from cloud '%2'").
			arg(cloudExtractions.size()).arg(cloud->getName()));

		if (error)
		{
			m_app->dispToConsole(
				tr("Error(s) occurred during the generation of segments! Result may be incomplete"),
				ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
		}

		cloud->setEnabled(false);
		if (cloud->getParent())
			cloud->getParent()->addChild(group);

		emit newEntity(group);

	}
	else if (error)
	{
		return -54;
	}
	else
	{
		return -1;
	}

	return 1;
}

// DGM this function will modify the cloudFiltered
int extractRecursive(
	PointCloudT::Ptr cloudFilterd,
	PointCloudT::Ptr cloudRemained,
	std::vector<PointCloudT::Ptr> &cloudExtractions,
	int maxIterations,
	float probability,
	float minRadiusLimits,
	float maxRadiusLimits,
	float distanceThreshold,
	int methodType,
	int modelType,
	float normalDisWeight,
	float maxRemainingRatio,
	bool exportExtraction,
	bool recursive,
	ecvMainAppInterface* app)
{
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

	int i = 0, nr_points = (int)cloudFilterd->points.size();
	while (cloudFilterd->points.size() > maxRemainingRatio * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		if (!PCLModules::GetSACSegmentation(
			cloudFilterd, inliers, coefficients, methodType, modelType,
			distanceThreshold, probability, maxIterations,
			minRadiusLimits, maxRadiusLimits, normalDisWeight))
		{
			app->dispToConsole("PCLModules::GetSACSegmentation failed",
								ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
			return -1;
		}

		if (inliers->indices.size() == 0)
		{
			break;
		}

		// Extract the extractions and remainings from the input cloud
		PointCloudT::Ptr cloudExtracted(new PointCloudT());
		if (!PCLModules::ExtractIndicesFilter<PointT>(
			cloudFilterd, inliers, cloudExtracted, cloudRemained))
		{
			app->dispToConsole("PCLModules::ExtractIndicesFilter failed",
				ecvMainAppInterface::WRN_CONSOLE_MESSAGE);
			return -1;
		}

		if (exportExtraction)
		{
			cloudExtractions.push_back(cloudExtracted);
		}
		*cloudFilterd = *cloudRemained;

		if (!recursive) break;
	}
	return 1;
}

QString SACSegmentation::getErrorMessage(int errorCode)
{
	switch(errorCode)
	{
		//THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS CODES
		//ALSO IN DERIVED CLASSES DEFULAT MUST BE ""

	case -51:
		return tr("must select one of 'Export Extraction' and 'exportRemaining' or both");
	case -52:
		return tr("Wrong Parameters. One or more parameters cannot be accepted");
	case -53:
		return tr("SAC Segmentation could not estimate a model for the given dataset. Try relaxing your parameters");
	case -54:
		return tr("An error occurred during the generation of segments!");
	}

	return BasePclModule::getErrorMessage(errorCode);
}
