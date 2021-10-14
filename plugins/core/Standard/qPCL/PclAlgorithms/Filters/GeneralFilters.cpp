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

#include "GeneralFilters.h"

// LOCAL
#include "dialogs/GeneralFiltersDlg.h"
#include "PclUtils/PCLModules.h"
#include "PclUtils/cc2sm.h"
#include "PclUtils/sm2cc.h"


// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// QT
#include <QMainWindow>

GeneralFilters::GeneralFilters()
	: BasePclModule(PclModuleDescription(tr("General Filters"),
										 tr("General Filters"),
										 tr("General Filters for the selected entity"),
										 ":/toolbar/PclAlgorithms/icons/generalFilter.png"))
    , m_dialog(nullptr)
	, m_polyline(nullptr)
	, m_dimension(2)
	, m_keepColors(true)
	, m_hasColors(false)
	, m_extractRemainings(false)
	, m_filterType(FilterType::CR_FILTER)
	, m_maxWindowSize(20)
	, m_slope(1.0f)
	, m_initialDistance(0.5f)
	, m_maxDistance(3.0f)
	, m_comparisonField("curvature")
	, m_comparisonTypes("GT")
	, m_minMagnitude(0.3f)
	, m_maxMagnitude(1.3f)
	, m_leafSize(0.01f)
{
}

GeneralFilters::~GeneralFilters()
{
	//we must delete parent-less dialogs ourselves!
    if (m_dialog && m_dialog->parent() == nullptr)
		delete m_dialog;
}

int GeneralFilters::openInputDialog()
{
	if (!m_dialog)
	{
		m_dialog = new GeneralFiltersDlg(m_app);
	}
	m_dialog->refreshPolylineComboBox();

	return m_dialog->exec() ? 1 : 0;
}

void GeneralFilters::getParametersFromDialog()
{
	assert(m_dialog);
	if (!m_dialog)
		return;

	// fill in parameters from dialog
	m_keepColors = m_dialog->keepColorCheckBox->isChecked();
	switch (m_dialog->tab->currentIndex())
	{
	// Cloud Pass Through Filter
	case 0:
	{
		m_comparisonField = m_dialog->getComparisonField(m_minMagnitude, m_maxMagnitude);
		m_filterType = FilterType::PASS_FILTER;
		break;
	}
	// Cloud Condition Removal Filter
	case 1:
	{
		m_comparisonField = m_dialog->getComparisonField(m_minMagnitude, m_maxMagnitude);
		m_dialog->getComparisonTypes(m_comparisonTypes);
		m_filterType = FilterType::CR_FILTER;
		break;
	}
	// Voxel Grid Filter
	case 2:
	{
		//m_sampleAllData = m_dialog->downsampleAllDataCheckBox->isChecked();
		m_leafSize = static_cast<float>(m_dialog->leafSizeSpinBox->value());
		m_filterType = FilterType::VOXEL_FILTER;
		break;
	}
	// Progressive Morphological Filter
	case 3:
	{
		bool m_extractRemainings = m_dialog->extractRemainingCheckBox->isChecked();
		int m_maxWindowSize = m_dialog->maxWindowSizeSpinBox->value();
		float m_slope = static_cast<float>(m_dialog->slopeSpinBox->value());
		float m_initialDistance = static_cast<float>(m_dialog->initialDistanceSpinBox->value());
		float m_maxDistance = static_cast<float>(m_dialog->maxDistanceSpinBox->value());
		m_filterType = FilterType::PM_FILTER;
		break;
	}
	// Crop Hull Filter
	case 4:
	{
		m_polyline = m_dialog->getPolyline();
		if (!m_polyline)
		{
			m_dialog->getContour(m_boundary);
		}
		
		m_dimension = m_dialog->dimensionSpinBox->value();
		m_filterType = FilterType::HULL_FILTER;
		break;
	}
	default:
		break;
	}

}

int GeneralFilters::checkParameters()
{
	if (m_minMagnitude > m_maxMagnitude)
	{
		return -52;
	}
	return 1;
}

int GeneralFilters::compute()
{
	// pointer to selected cloud
	ccPointCloud* cloud = getSelectedEntityAsCCPointCloud();
	if (!cloud)
		return -1;

	// initialize all possible clouds
	PCLCloud::Ptr out_cloud_sm(new PCLCloud);
	PointCloudT::Ptr xyzCloud(new PointCloudT);
	PointCloudRGB::Ptr rgbCloud(new PointCloudRGB);

	PCLCloud::Ptr sm_cloud = cc2smReader(cloud).getAsSM();
	if (cloud->hasColors() || cloud->hasScalarFields())
	{
		FROM_PCL_CLOUD(*sm_cloud, *rgbCloud);
		if (!rgbCloud)
			return -1;
		m_hasColors = true;
	}
	else
	{
		FROM_PCL_CLOUD(*sm_cloud, *xyzCloud);
		if (!xyzCloud)
			return -1;
		m_hasColors = false;
	}


	QString name = tr("%1-out").arg(cloud->getName());
	switch (m_filterType)
	{
	case GeneralFilters::PASS_FILTER:
	{
		if (m_hasColors)
		{
			PointCloudRGB::Ptr tempCloud(new PointCloudRGB);
			if (!PCLModules::PassThroughFilter<PointRGB>(rgbCloud, tempCloud,
				m_comparisonField, m_minMagnitude, m_maxMagnitude))
			{
				return -1;
			}
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}
		else
		{
			PointCloudT::Ptr tempCloud(new PointCloudT);
			if (!PCLModules::PassThroughFilter<PointT>(xyzCloud, tempCloud,
				m_comparisonField, m_minMagnitude, m_maxMagnitude))
			{
				return -1;
			}
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}
		name = tr("%1-passThrough").arg(cloud->getName());
		break;
	}
	case GeneralFilters::CR_FILTER:
	{
		// init condition parameters
		PCLModules::ConditionParameters param;
		{
			if (m_comparisonTypes.size() == 1)
			{
				param.condition_type_ = PCLModules::ConditionParameters::ConditionType::CONDITION_OR;
			}
			else
			{
				param.condition_type_ = PCLModules::ConditionParameters::ConditionType::CONDITION_AND;
			}
			for (const QString& type : m_comparisonTypes)
			{
				PCLModules::ConditionParameters::ComparisonParam comparison;
				if (type == "GT")
				{
					comparison.comparison_type_ = PCLModules::ConditionParameters::ComparisonType::GT;
				}
				else if (type == "GE")
				{
					comparison.comparison_type_ = PCLModules::ConditionParameters::ComparisonType::GE;
				}
				else if (type == "LT")
				{
					comparison.comparison_type_ = PCLModules::ConditionParameters::ComparisonType::LT;
				}
				else if (type == "LE")
				{
					comparison.comparison_type_ = PCLModules::ConditionParameters::ComparisonType::LE;
				}
				else if (type == "EQ")
				{
					comparison.comparison_type_ = PCLModules::ConditionParameters::ComparisonType::EQ;
				}
				comparison.fieldName_ = m_comparisonField.toStdString();
				comparison.min_threshold_ = m_minMagnitude;
				comparison.max_threshold_ = m_maxMagnitude;
				param.condition_params_.push_back(comparison);
			}
		}

		if (m_hasColors)
		{
			PointCloudRGB::Ptr tempCloud(new PointCloudRGB);
			if (!PCLModules::ConditionalRemovalFilter<PointRGB>(
				rgbCloud, param, tempCloud, true))
			{
				return -1;
			}
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}
		else
		{
			PointCloudT::Ptr tempCloud(new PointCloudT);
			if (!PCLModules::ConditionalRemovalFilter<PointT>(
				xyzCloud, param, tempCloud, true))
			{
				return -1;
			}
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}
		name = tr("%1-condition").arg(cloud->getName());
		break;
	}
	case GeneralFilters::VOXEL_FILTER:
	{
		if (m_hasColors)
		{
			PointCloudRGB::Ptr tempCloud(new PointCloudRGB);
			if (!PCLModules::VoxelGridFilter<PointRGB>(rgbCloud, tempCloud,
				m_leafSize, m_leafSize, m_leafSize))
			{
				return -1;
			}

			// if voxel grid failed and just return ignore this operation
			if (tempCloud->size() == rgbCloud->size())
			{
				return 1;
			}
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}
		else
		{
			PointCloudT::Ptr tempCloud(new PointCloudT);
			if (!PCLModules::VoxelGridFilter<PointT>(xyzCloud, tempCloud,
				m_leafSize, m_leafSize, m_leafSize))
			{
				return -1;
			}
			// if voxel grid failed and just return ignore this operation
			if (tempCloud->size() == xyzCloud->size())
			{
				return 1;
			}
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}

		name = tr("%1-voxelGrid").arg(cloud->getName());
		break;
	}
	case GeneralFilters::PM_FILTER:
	{
		if (m_hasColors)
		{
			pcl::PointIndicesPtr ground (new pcl::PointIndices);
			if (!PCLModules::ProgressiveMpFilter<PointRGB>(rgbCloud, ground,
				m_maxWindowSize, m_slope, m_initialDistance, m_maxDistance) ||
				ground->indices.size() == 0)
			{
				return -1;
			}

			PointCloudRGB::Ptr tempCloud(new PointCloudRGB);
			if (m_extractRemainings)
			{
				if (!PCLModules::ExtractIndicesFilter<PointRGB>(rgbCloud, ground, nullptr, tempCloud))
				{
					return -1;
				}
			}
			else
			{
				if (!PCLModules::ExtractIndicesFilter<PointRGB>(rgbCloud, ground, tempCloud, nullptr))
				{
					return -1;
				}
			}
			
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}
		else
		{
			pcl::PointIndicesPtr ground(new pcl::PointIndices);
			if (!PCLModules::ProgressiveMpFilter<PointT>(xyzCloud, ground,
				m_maxWindowSize, m_slope, m_initialDistance, m_maxDistance) ||
				ground->indices.size() == 0)
			{
				return -1;
			}

			PointCloudT::Ptr tempCloud(new PointCloudT);
			if (m_extractRemainings)
			{
				if (!PCLModules::ExtractIndicesFilter<PointT>(xyzCloud, ground, nullptr, tempCloud))
				{
					return -1;
				}
			}
			else
			{
				if (!PCLModules::ExtractIndicesFilter<PointT>(xyzCloud, ground, tempCloud, nullptr))
				{
					return -1;
				}
			}

			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}
		name = tr("%1-progressive").arg(cloud->getName());
		break;
	}
	case GeneralFilters::HULL_FILTER:
	{
		PointCloudT::Ptr boundary(new PointCloudT);
		if (m_polyline && m_polyline->size() > 2)
		{
			unsigned int pointNum = m_polyline->size();
			boundary->resize(pointNum);
			for (unsigned int i = 0; i < pointNum; ++i)
			{
				const CCVector3* p = m_polyline->getPoint(i);
				boundary->points[i].x = p->x;
				boundary->points[i].y = p->y;
				boundary->points[i].z = p->z;
			}
		}
		else
		{
			size_t pointNum = m_boundary.size();
			boundary->resize(pointNum);
			for (size_t i = 0; i < pointNum; ++i)
			{
				boundary->points[i].x = m_boundary[i].x;
				boundary->points[i].y = m_boundary[i].y;
				boundary->points[i].z = m_boundary[i].z;
			}
		}

		if (boundary->width * boundary->height == 0)
		{
			return -1;
		}
		
		if (m_hasColors)
		{
			PointCloudRGB::Ptr tempCloud(new PointCloudRGB);
			if (!PCLModules::CropHullFilter<PointRGB>(rgbCloud, boundary, tempCloud, m_dimension))
			{
				return -1;
			}
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}
		else
		{
			PointCloudT::Ptr tempCloud(new PointCloudT);
			if (!PCLModules::CropHullFilter<PointT>(xyzCloud, boundary, tempCloud, m_dimension))
			{
				return -1;
			}
			TO_PCL_CLOUD(*tempCloud, *out_cloud_sm);
		}

		name = tr("%1-cropHull").arg(cloud->getName());
		break;
	}

	default:
		break;
	}

	if (out_cloud_sm->width * out_cloud_sm->height == 0)
	{
		return -53;
	}
	
    ccPointCloud* out_cloud_cc = pcl2cc::Convert(*out_cloud_sm);
	{
		if (!out_cloud_cc)
		{
			//conversion failed (not enough memory?)
			return -1;
		}
		out_cloud_cc->setName(name);
		if (!m_keepColors)
		{
			ecvColor::Rgb col = ecvColor::Generator::Random();
			out_cloud_cc->setRGBColor(col);
			out_cloud_cc->showColors(true);
			out_cloud_cc->showSF(false);
		}

		// copy global shift & scale
		out_cloud_cc->setGlobalScale(cloud->getGlobalScale());
		out_cloud_cc->setGlobalShift(cloud->getGlobalShift());
	}

	cloud->setEnabled(false);
	if (cloud->getParent())
		cloud->getParent()->addChild(out_cloud_cc);

	emit newEntity(out_cloud_cc);

	return 1;
}


QString GeneralFilters::getErrorMessage(int errorCode)
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
		return tr("General Filter does not returned any point. Try relaxing your parameters");
	}

	return BasePclModule::getErrorMessage(errorCode);
}
