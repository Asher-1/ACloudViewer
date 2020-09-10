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
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//

#include "ProjectionFilter.h"

// LOCAL
#include "dialogs/ProjectionFilterDlg.h"
#include "PclUtils/PCLModules.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"


// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>

// QT
#include <QMainWindow>

ProjectionFilter::ProjectionFilter()
	: BasePclModule(PclModuleDescription(tr("Projection Filter"),
										 tr("Projection Filter"),
										 tr("Projection Filter for the selected entity"),
										 ":/toolbar/PclAlgorithms/icons/filter_projection.png"))
	, m_dialog(0)
	, m_projectionMode(true)
	, m_coefficientA(0.0f)
	, m_coefficientB(0.0f)
	, m_coefficientC(1.0f)
	, m_coefficientD(0.0f)
	, m_useVoxelGrid(false)
	, m_leafSize(0.01f)
	, m_useKnn(true)
	, m_knn_radius(20)
	, m_normalSearchRadius(0.01f)
	, m_boundarySearchRadius(0.05f)
	, m_boundaryAngleThreshold(90.0f)
{
}

ProjectionFilter::~ProjectionFilter()
{
	//we must delete parent-less dialogs ourselves!
	if (m_dialog && m_dialog->parent() == 0)
		delete m_dialog;
}

int ProjectionFilter::openInputDialog()
{
	if (!m_dialog)
	{
		m_dialog = new ProjectionFilterDlg(m_app ? m_app->getActiveWindow() : 0);
	}

	return m_dialog->exec() ? 1 : 0;
}

void ProjectionFilter::getParametersFromDialog()
{
	assert(m_dialog);
	if (!m_dialog)
		return;

	switch (m_dialog->tab->currentIndex())
	{
		// Cloud Projection Filter
	case 0:
	{
		// fill in parameters from dialog
		m_coefficientA = static_cast<float>(m_dialog->aSpinBox->value());
		m_coefficientB = static_cast<float>(m_dialog->bSpinBox->value());
		m_coefficientC = static_cast<float>(m_dialog->cSpinBox->value());
		m_coefficientD = static_cast<float>(m_dialog->dSpinBox->value());
		m_projectionMode = true;
	}
	break;
	// Cloud Boundary Filter
	case 1:
	{
		m_useVoxelGrid = m_dialog->useVoxelGridCheckBox->isChecked();
		m_leafSize = static_cast<float>(m_dialog->leafSizeSpinBox->value());
		m_useKnn = m_dialog->useKnnBoundaryCheckBox->isChecked();
		m_knn_radius = m_dialog->knnrBoundarySpinBox->value();
		m_normalSearchRadius = static_cast<float>(m_dialog->normalSearchRadius->value());

		m_boundaryAngleThreshold = static_cast<float>(m_dialog->boundaryAngleThresholdSpinBox->value());
		m_projectionMode = false;

	}
	break;
	}

}

int ProjectionFilter::checkParameters()
{
	return 1;
}

int ProjectionFilter::compute()
{
	// pointer to selected cloud
	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (!cloud)
		return -1;

	PointCloudT::Ptr xyzCloud = cc2smReader(cloud).getXYZ2();
	if (!xyzCloud)
		return -1;

	// initialize all possible clouds
	PCLCloud::Ptr out_cloud_sm(new PCLCloud);
	PointCloudT::Ptr outCloudxyz(new PointCloudT);


	QString name;
	if (m_projectionMode) // Projection mode
	{
		// originCloud: cloud before projection
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
		coefficients->values.resize(4);
		coefficients->values[0] = m_coefficientA;
		coefficients->values[1] = m_coefficientB;
		coefficients->values[2] = m_coefficientC;
		coefficients->values[3] = m_coefficientD;

		if (!PCLModules::GetProjection(xyzCloud, outCloudxyz, coefficients, 0))
			return -1;
		name = tr("%1-projection").arg(cloud->getName());
	}
	else // boundary mode
	{
		// voxel grid filter
		if (m_useVoxelGrid)
		{
			PointCloudT::Ptr tempCloud(new PointCloudT);
			if (!PCLModules::VoxelGridFilter<PointT>(xyzCloud, tempCloud,
				m_leafSize, m_leafSize, m_leafSize))
			{
				return -1;
			}
			xyzCloud = tempCloud;
		}

		// create storage for normals
		pcl::PointCloud<NormalT>::Ptr normals(new pcl::PointCloud<NormalT>);
		{
			if (!cloud->hasNormals())
			{
				// now compute normals
				int result = PCLModules::ComputeNormals<PointT, NormalT>(
					xyzCloud, normals, m_useKnn ? m_knn_radius : m_normalSearchRadius, m_useKnn);
				if (result < 0)
					return -1;
			}
			else
			{
				PCLCloud::Ptr sm_cloud = cc2smReader(cloud).getNormals();
				FROM_PCL_CLOUD(*sm_cloud, *normals);
			}
		}
		
		if (!m_useKnn)
		{
			float resolution = 
				static_cast<float>(PCLModules::ComputeCloudResolution<PointT>(xyzCloud));
			if (resolution > 0)
			{
				m_boundarySearchRadius = 10 * resolution;
			}
		}

		if (!PCLModules::GetBoundaryCloud<PointT, NormalT>(
			xyzCloud, normals, outCloudxyz, m_boundaryAngleThreshold,
			m_useKnn ? m_knn_radius : m_boundarySearchRadius, m_useKnn))
			return -1;

		name = tr("%1-boundary").arg(cloud->getName());
	}

	if (outCloudxyz->width * outCloudxyz->height == 0)
	{
		return -53;
	}

	TO_PCL_CLOUD(*outCloudxyz, *out_cloud_sm);
	ccPointCloud* out_cloud_cc = sm2ccConverter(out_cloud_sm).getCloud();
	{
		if (!out_cloud_cc)
		{
			//conversion failed (not enough memory?)
			return -1;
		}
		out_cloud_cc->setName(name);
		ecvColor::Rgb col = ecvColor::Generator::Random();
		out_cloud_cc->setRGBColor(col);
		out_cloud_cc->showColors(true);
		out_cloud_cc->showSF(false);

		//copy global shift & scale
		out_cloud_cc->setGlobalScale(cloud->getGlobalScale());
		out_cloud_cc->setGlobalShift(cloud->getGlobalShift());
	}

	if (cloud->getParent())
		cloud->getParent()->addChild(out_cloud_cc);

	emit newEntity(out_cloud_cc);

	return 1;
}


QString ProjectionFilter::getErrorMessage(int errorCode)
{
	switch (errorCode)
	{
		//THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS CODES
		//ALSO IN DERIVED CLASSES DEFULAT MUST BE ""

	case -51:
		return tr("Selected entity does not have any suitable scalar field or RGB. Intensity scalar field or RGB are needed for computing SIFT");
	case -52:
		return tr("Wrong Parameters. One or more parameters cannot be accepted");
	case -53:
		return tr("Projection extraction does not returned any point. Try relaxing your parameters");
	}

	return BasePclModule::getErrorMessage(errorCode);
}