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
#include "GreedyTriangulation.h"
#include "dialogs/GreedyTriangulationDlg.h"
#include "PclUtils/PCLModules.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// QT
#include <QMainWindow>

// SYSTEM
#include <iostream>

GreedyTriangulation::GreedyTriangulation()
	: BasePclModule(PclModuleDescription(tr("Greedy Triangulation"),
										 tr("Greedy Triangulation"),
										 tr("Greedy Triangulation from clouds"),
										 ":/toolbar/PclAlgorithms/icons/greedy_triangulation.png"))
	, m_dialog(nullptr)
	, m_normalSearchRadius(10)
	, m_knn_radius(10)
	, m_useKnn(false)
	, m_trigulationSearchRadius(25)
	, m_weightingFactor(2.5)
	, m_maxNearestNeighbors(100)
	, m_maxSurfaceAngle(45)
	, m_minAngle(10)
	, m_maxAngle(120)
	, m_normalConsistency(false)
{
}

GreedyTriangulation::~GreedyTriangulation()
{
	//we must delete parent-less dialogs ourselves!
	if (m_dialog && m_dialog->parent() == nullptr)
		delete m_dialog;
}

int GreedyTriangulation::checkSelected()
{
	//do we have a selected cloud?
	int have_cloud = isFirstSelectedCcPointCloud();
	if (have_cloud != 1)
		return -11;

	return 1;
}

int GreedyTriangulation::openInputDialog()
{
	//initialize the dialog object
	if (!m_dialog)
		m_dialog = new GreedyTriangulationDlg(m_app ? m_app->getActiveWindow() : 0);

	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (cloud)
	{
		ccBBox bBox = cloud->getOwnBB();
		if (bBox.isValid())
			m_dialog->normalSearchRadius->setValue(bBox.getDiagNorm() * 0.005);
	}

	if (!m_dialog->exec())
		return 0;

	return 1;
}

void GreedyTriangulation::getParametersFromDialog()
{
	if (!m_dialog)
		return;

	//get the parameters from the dialog
	m_normalConsistency = m_dialog->normalConsistency->isChecked();
	m_useKnn = m_dialog->useKnnCheckBoxForTriangulation->isChecked();
	m_knn_radius = m_dialog->knnSpinBoxForTriangulation->value();
	m_normalSearchRadius = static_cast<float>(m_dialog->normalSearchRadius->value());

	m_trigulationSearchRadius = m_dialog->triangulationSearchRadius->value();
	m_weightingFactor = static_cast<float>(m_dialog->weightingFactor->value());
	m_maxNearestNeighbors = m_dialog->nearestThreshold->value();
	m_maxSurfaceAngle = m_dialog->maxSurfaceAngle->value();
	m_minAngle = m_dialog->minAngle->value();
	m_maxAngle = m_dialog->maxAngle->value();
}

int GreedyTriangulation::checkParameters()
{
	if (m_maxAngle > m_minAngle)
	{
		return 1;
	}
	else
	{
		return -52;
	}
}

int GreedyTriangulation::compute()
{
	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (!cloud)
		return -1;

	// create storage for normals
	pcl::PointCloud<pcl::PointNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointNormal>);

	if (!cloud->hasNormals())
	{
		// get xyz as pcl point cloud
		pcl::PointCloud<pcl::PointXYZ>::Ptr xyzCloud = cc2smReader(cloud).getXYZ2();
		if (!xyzCloud)
			return -1;

		// now compute
		pcl::PointCloud<NormalT>::Ptr normals(new pcl::PointCloud<NormalT>);
		int result = PCLModules::ComputeNormals<pcl::PointXYZ, NormalT>(
			xyzCloud, normals, m_useKnn ? m_knn_radius : m_normalSearchRadius, m_useKnn);
		if (result < 0)
			return -1;

		// concat points and normals
		pcl::concatenateFields(*xyzCloud, *normals, *cloudWithNormals);
		CVLog::Print(tr("[GreedyTriangulation::compute] generate new normals"));
	} 
	else
	{
		PCLCloud::Ptr sm_cloud = cc2smReader(cloud).getAsSM();
		FROM_PCL_CLOUD(*sm_cloud, *cloudWithNormals);
		CVLog::Print(tr("[GreedyTriangulation::compute] find normals and use the normals"));
	}

	// reconstruction
	PCLMesh mesh;
	int result = PCLModules::GetGreedyTriangulation(
							cloudWithNormals, mesh,
							m_trigulationSearchRadius, m_weightingFactor,
							m_maxNearestNeighbors, m_maxSurfaceAngle,
							m_minAngle, m_maxAngle, m_normalConsistency);
	if (result < 0)
		return -1;

	PCLCloud::Ptr out_cloud_sm (new PCLCloud(mesh.cloud));
	if ( out_cloud_sm->height * out_cloud_sm->width == 0)
	{
		//cloud is empty
		return -53;
	}

	ccMesh* out_mesh = sm2ccConverter(out_cloud_sm).getMesh(mesh.polygons);
	if (!out_mesh)
	{
		//conversion failed (not enough memory?)
		return -1;
	}

	unsigned vertCount = out_mesh->getAssociatedCloud()->size();
	unsigned faceCount = out_mesh->size();
	CVLog::Print(tr("[greedy-triangulation-Reconstruction] %1 points, %2 face(s)").arg(vertCount).arg(faceCount));

	out_mesh->setName(tr("greedy-triangulation searchRadius[%1]").arg(m_trigulationSearchRadius));
	//copy global shift & scale
	out_mesh->getAssociatedCloud()->setGlobalScale(cloud->getGlobalScale());
	out_mesh->getAssociatedCloud()->setGlobalShift(cloud->getGlobalShift());

	if (cloud->getParent())
		cloud->getParent()->addChild(out_mesh);

	emit newEntity(out_mesh);

	return 1;
}

QString GreedyTriangulation::getErrorMessage(int errorCode)
{
	switch(errorCode)
	{
		//THESE CASES CAN BE USED TO OVERRIDE OR ADD FILTER-SPECIFIC ERRORS CODES
		//ALSO IN DERIVED CLASSES DEFULAT MUST BE ""

	case -51:
		return tr("Selected entity does not have any suitable scalar field or RGB.");
	case -52:
		return tr("Wrong Parameters. One or more parameters cannot be accepted");
	case -53:
		return tr("Greedy Triangulation does not returned any point. Try relaxing your parameters");
	}

	return BasePclModule::getErrorMessage(errorCode);
}
