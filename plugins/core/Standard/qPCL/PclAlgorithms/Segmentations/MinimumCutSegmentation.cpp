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
#include "MinimumCutSegmentation.h"
#include "dialogs/MinimumCutSegmentationDlg.h"
#include "PclUtils/PCLModules.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"

// ECV_DB_LIB
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>


// SYSTEM
#include <iostream>
#include <sstream>

MinimumCutSegmentation::MinimumCutSegmentation()
	: BasePclModule(PclModuleDescription(tr("Min Cut Segmentation"),
										 tr("Min Cut Segmentation"),
										 tr("Min Cut Segmentation from clouds"),
										 ":/toolbar/PclAlgorithms/icons/mincut.png"))
	, m_dialog(nullptr)
	, m_colored(false)
	, m_neighboursNumber(14)
	, m_smoothSigma(0.25f)
	, m_backWeightRadius(3.0f)
	, m_foregroundWeight(0.8f)
	, m_cx(0.0f)
	, m_cy(0.0f)
	, m_cz(0.0f)
{
}

MinimumCutSegmentation::~MinimumCutSegmentation()
{
	//we must delete parent-less dialogs ourselves!
	if (m_dialog && m_dialog->parent() == nullptr)
		delete m_dialog;
}

int MinimumCutSegmentation::checkSelected()
{
	//do we have a selected cloud?
	int have_cloud = isFirstSelectedCcPointCloud();
	if (have_cloud != 1)
		return -11;

	return 1;
}

int MinimumCutSegmentation::openInputDialog()
{
	//initialize the dialog object
	if (!m_dialog)
		m_dialog = new MinimumCutSegmentationDlg(m_app);

	m_dialog->refreshLabelComboBox();
	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (cloud->hasColors() || cloud->hasScalarFields())
	{
		m_colored = true;
	}
	else
	{
		m_colored = false;
	}

	if (!m_dialog->exec())
		return 0;

	return 1;
}

void MinimumCutSegmentation::getParametersFromDialog()
{
	if (!m_dialog)
		return;

	m_cx = static_cast<float>(m_dialog->cxAxisDoubleSpinBox->value());
	m_cy = static_cast<float>(m_dialog->cyAxisDoubleSpinBox->value());
	m_cz = static_cast<float>(m_dialog->czAxisDoubleSpinBox->value());

	m_neighboursNumber = m_dialog->neighboursNumSpinBox->value();
	m_smoothSigma = static_cast<float>(m_dialog->smoothSigmaSpinbox->value());
	m_backWeightRadius = static_cast<float>(m_dialog->backWeightRadiusSpinbox->value());
	m_foregroundWeight = static_cast<float>(m_dialog->foreWeightSpinbox->value());

}

int MinimumCutSegmentation::checkParameters()
{
	return 1;
}

int MinimumCutSegmentation::compute()
{
	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (!cloud)
		return -1;

	PCLCloud::Ptr sm_cloud = cc2smReader(cloud).getAsSM();
	if (!sm_cloud)
		return -1;

	// initialize all possible clouds
	std::vector<pcl::PointIndices> clusters;
	PointCloudT::Ptr xyzCloud(new PointCloudT);
	PointCloudRGB::Ptr rgbCloud(new PointCloudRGB);
	PointCloudRGB::Ptr cloudSegmented(new PointCloudRGB);

	if (m_colored) // XYZRGB
	{
		PointRGB foregroundPoint(0.0f, 0.0f, 0.0f, 255, 255, 255);
		foregroundPoint.x = m_cx;
		foregroundPoint.y = m_cy;
		foregroundPoint.z = m_cz;
		FROM_PCL_CLOUD(*sm_cloud, *rgbCloud);
		int result = PCLModules::GetMinCutSegmentation<PointRGB>(
			rgbCloud, clusters, cloudSegmented, foregroundPoint,
			m_neighboursNumber, m_smoothSigma,
			m_backWeightRadius, m_foregroundWeight);
		if (result < 0)
			return -1;
	}
	else // XYZ
	{
		FROM_PCL_CLOUD(*sm_cloud, *xyzCloud);
		PointT foregroundPoint(m_cx, m_cy, m_cz);
		int result = PCLModules::GetMinCutSegmentation<PointT>(
			xyzCloud, clusters, cloudSegmented, foregroundPoint,
			m_neighboursNumber, m_smoothSigma,
			m_backWeightRadius, m_foregroundWeight);
		if (result < 0)
			return -1;
	}

    PCLCloud out_cloud_sm;
    TO_PCL_CLOUD(*cloudSegmented, out_cloud_sm);

    if (out_cloud_sm.height * out_cloud_sm.width == 0)
	{
		//cloud is empty
		return -53;
	}

    ccPointCloud* out_cloud_cc = pcl2cc::Convert(out_cloud_sm);
	if (!out_cloud_cc)
	{
		//conversion failed (not enough memory?)
		return -1;
	}

	if (cloud)
	{
		out_cloud_cc->setName(cloud->getName() + "-min-cut");
		//copy global shift & scale
		out_cloud_cc->setGlobalScale(cloud->getGlobalScale());
		out_cloud_cc->setGlobalShift(cloud->getGlobalShift());

		cloud->setEnabled(false);
		if (cloud->getParent())
			cloud->getParent()->addChild(out_cloud_cc);
	}
	emit newEntity(out_cloud_cc);

	return 1;
}

QString MinimumCutSegmentation::getErrorMessage(int errorCode)
{
	switch(errorCode)
	{
		//THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS CODES
		//ALSO IN DERIVED CLASSES DEFULAT MUST BE ""

	case -51:
		return tr("Selected entity does not have any suitable scalar field or RGB");
	case -52:
		return tr("Wrong Parameters. One or more parameters cannot be accepted");
	case -53:
		return tr("Min Cut Segmentation does not returned any point. Try relaxing your parameters");
	}

	return BasePclModule::getErrorMessage(errorCode);
}
