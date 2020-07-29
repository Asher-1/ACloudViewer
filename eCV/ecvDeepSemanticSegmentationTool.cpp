//##########################################################################
//#                                                                        #
//#                            CLOUDVIEWER                                 #
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

#include "ecvDeepSemanticSegmentationTool.h"
#include "ecvEntityAction.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>

// ECV_DB_LIB
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvDisplayTools.h>

// Qt
#include <QFuture>
#include <QApplication>
#include <QProgressDialog>
#include <qtconcurrentrun.h>

// System
#if defined(CV_WINDOWS)
#include "windows.h"
#else
#include <time.h>
#include <unistd.h>
#endif

#ifdef ECV_PYTHON_USE_AS_DLL
ecvDeepSemanticSegmentationTool::ecvDeepSemanticSegmentationTool(QWidget* parent)
	: ccOverlayDialog(parent)
	, Ui::DeepSemanticSegmentationDlg()
	, m_show_progress(true)
{
	setupUi(this);
	connect(detectToolButton, &QAbstractButton::clicked, this, &ecvDeepSemanticSegmentationTool::detect);
	connect(appyToolButton, &QAbstractButton::clicked, this, &ecvDeepSemanticSegmentationTool::apply);
	connect(cancelToolButton, &QAbstractButton::clicked, this, &ecvDeepSemanticSegmentationTool::cancel);
	connect(selectAllRadioButton, &QAbstractButton::clicked, this, &ecvDeepSemanticSegmentationTool::selectAllClasses);
	connect(unselectAllRadioButton, &QAbstractButton::clicked, this, &ecvDeepSemanticSegmentationTool::selectAllClasses);

}

ecvDeepSemanticSegmentationTool::~ecvDeepSemanticSegmentationTool()
{
}

bool ecvDeepSemanticSegmentationTool::linkWith(QWidget* win)
{
	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}

	return true;
}

bool ecvDeepSemanticSegmentationTool::addEntity(ccHObject * entity)
{
	if (!entity || !entity->isKindOf(CV_TYPES::POINT_CLOUD))
		return false;

	int pointNumber = static_cast<int>(ccHObjectCaster::ToPointCloud(entity)->size());
	if (pointNumber < 65536)
	{
		CVLog::Warning(
			QString("[ecvDeepSemanticSegmentationTool::addEntity] " 
				"Skip entity [%1] as the point number of it is %2 lower than min limit 65536!")
		.arg(entity->getName(), pointNumber));
		return false;
	}

	m_entity.addChild(entity, ccHObject::DP_NONE);

	return true;
}

unsigned ecvDeepSemanticSegmentationTool::getNumberOfValidEntities() const
{
	return m_entity.getChildrenNumber();
}

bool ecvDeepSemanticSegmentationTool::start()
{
	if (!ecvDisplayTools::GetCurrentScreen())
		return false;

	// clear history in case
	if (!m_clusters_map.empty())
	{
		m_clusters_map.clear();
	}
	if (!m_clusters.empty())
	{
		m_clusters.clear();
	}

	if (m_selectedEntity.getChildrenNumber() != 0)
	{
		m_selectedEntity.detatchAllChildren();
	}

	unsigned childNum = getNumberOfValidEntities();
	if (childNum == 0)
		return false;

	selectedTreeWiget->headerItem()->setCheckState(0, Qt::Checked);
	for (unsigned i = 0; i < getNumberOfValidEntities(); ++i)
	{
		QTreeWidgetItem * item = new QTreeWidgetItem();
		item->setCheckState(0, Qt::Checked);
		item->setText(1, m_entity.getChild(i)->getName());
		selectedTreeWiget->insertTopLevelItem(i, item);
	}

#ifdef ECV_PYTHON_USE_AS_DLL
	return ccOverlayDialog::start();
#else
	return false;
#endif

}

void ecvDeepSemanticSegmentationTool::apply()
{
	if (m_clusters_map.empty()) // no preview and directly apply
	{
		performSegmentation();
	}
	stop(true);
	clear();
	return;
}

void ecvDeepSemanticSegmentationTool::detect()
{
	if (performSegmentation() > 0)
	{
		exportClustersToSF();
		refreshSelectedClouds();
	}
}

void ecvDeepSemanticSegmentationTool::refreshSelectedClouds()
{
	ecvDisplayTools::SetRedrawRecursive(false);
	for (unsigned i = 0; i < m_selectedEntity.getChildrenNumber(); ++i)
	{
		m_selectedEntity.getChild(i)->setRedrawFlagRecursive(true);
	}
	ecvDisplayTools::RedrawDisplay();
}

void ecvDeepSemanticSegmentationTool::cancel()
{
	stop(false);
	clear();
}

void ecvDeepSemanticSegmentationTool::stop(bool state)
{
	ccOverlayDialog::stop(state);
}

void ecvDeepSemanticSegmentationTool::clear()
{
	m_entity.detatchAllChildren();
	for (int i = 0; i < selectedTreeWiget->topLevelItemCount(); ++i) {
		delete selectedTreeWiget->takeTopLevelItem(i);
	}

	m_clusters_map.clear();
	m_clusters.clear();
	m_selectedEntity.detatchAllChildren();
}

void ecvDeepSemanticSegmentationTool::getSegmentations(ccHObject::Container & result)
{
	if (exportModeComboBox->currentIndex() == 0)
	{
		exportClustersToEntities(result);
	}
	else if (exportModeComboBox->currentIndex() == 1)
	{
		exportClustersToSF();
	}
}

void ecvDeepSemanticSegmentationTool::updateSelectedEntity()
{
	m_selectedEntity.detatchAllChildren();
	for (int i = 0; i < selectedTreeWiget->topLevelItemCount(); ++i) 
	{
		QTreeWidgetItem * item = selectedTreeWiget->topLevelItem(i);
		if (item->checkState(0) == Qt::Checked)
		{
			for (unsigned j = 0; j < getNumberOfValidEntities(); ++j)
			{
				if (item->text(1) == m_entity.getChild(j)->getName())
				{
					m_selectedEntity.addChild(m_entity.getChild(j), ccHObject::DP_NONE);
					break;
				}
			}
		}
	}
}

void ecvDeepSemanticSegmentationTool::exportClustersToSF()
{
	ccHObject::Container selectedClouds;
	m_selectedEntity.filterChildren(selectedClouds, false, CV_TYPES::POINT_CLOUD);
	if (m_clusters.size() != selectedClouds.size())
	{
		CVLog::Error("[ecvDeepSemanticSegmentationTool] dimensions do not match!");
		return;
	}

	std::vector< std::vector<ScalarType> > scalarsVector;
	ccEntityAction::ConvertToScalarType<size_t>(m_clusters, scalarsVector);
	if (!ccEntityAction::importToSF(selectedClouds, scalarsVector, "Clusters"))
		CVLog::Error("[ecvDeepSemanticSegmentationTool::exportClustersToSF] import sf failed!");
}

void ecvDeepSemanticSegmentationTool::exportClustersToEntities(ccHObject::Container & result)
{
	bool needFresh = false;
	for (unsigned int i = 0; i < m_selectedEntity.getChildrenNumber(); ++i)
	{
		ccHObject* ent = m_selectedEntity.getChild(i);
		if (!ent)
		{
			continue;
		}

		ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent);
		if (!cloud)
		{
			continue;
		}

		// remove preview scalar field if exist
		if (cloud->hasDisplayedScalarField())
		{
			needFresh = true;
			cloud->deleteScalarField(cloud->getCurrentDisplayedScalarFieldIndex());
			cloud->setCurrentDisplayedScalarField(-1);
			cloud->showSF(false);
		}

		// we create a new group to store all input CCs as 'clusters'
		ccHObject* ecvGroup = new ccHObject(ent->getName() + "-recognition");
		ecvGroup->setVisible(true);

		const ClassMap::ClusterMap& clusterMap = m_clusters_map[i];
		for (ClassMap::ClusterMap::const_iterator it = clusterMap.begin();
			it != clusterMap.end(); ++it)
		{
			ccPointCloud* res = ccPointCloud::From(cloud, it->second);
			if (res)
			{
				res->setName(it->first.c_str());
				ecvGroup->addChild(res);
			}
		}

		if (ecvGroup->getChildrenNumber() == 0)
		{
			delete ecvGroup;
			ecvGroup = nullptr;
		}
		else
		{
			if (ent->getParent())
				ent->getParent()->addChild(ecvGroup);
			result.push_back(ecvGroup);
		}
	}

	if (needFresh)
	{
		refreshSelectedClouds();
	}

	for (unsigned i = 0; i < m_selectedEntity.getChildrenNumber(); ++i)
	{
		m_selectedEntity.getChild(i)->setEnabled(false);
	}
}

void ecvDeepSemanticSegmentationTool::getSelectedFilterClasses(std::vector<size_t>& filteredClasses)
{
	if (!filteredClasses.empty())
	{
		filteredClasses.clear();
	}

	if (unlabeledCheckBox->isChecked())
		filteredClasses.push_back(0);
	if (manMadeTerrainCheckBox->isChecked())
		filteredClasses.push_back(1);
	if (naturalTerrainCheckBox->isChecked())
		filteredClasses.push_back(2);
	if (highVegetationCheckBox->isChecked())
		filteredClasses.push_back(3);
	if (lowVegetationCheckBox->isChecked())
		filteredClasses.push_back(4);
	if (buildingsCheckBox->isChecked())
		filteredClasses.push_back(5);
	if (hardScapeCheckBox->isChecked())
		filteredClasses.push_back(6);
	if (scanningArtifactsCheckBox->isChecked())
		filteredClasses.push_back(7);
	if (carsCheckBox->isChecked())
		filteredClasses.push_back(8);
	if (utilityPoleCheckBox->isChecked())
		filteredClasses.push_back(9);
	if (insulatorCheckBox->isChecked())
		filteredClasses.push_back(10);
	if (electricalWireCheckBox->isChecked())
		filteredClasses.push_back(11);
	if (crossBarCheckBox->isChecked())
		filteredClasses.push_back(12);
	if (stickCheckBox->isChecked())
		filteredClasses.push_back(13);
	if (fuseCheckBox->isChecked())
		filteredClasses.push_back(14);
	if (wireClipCheckBox->isChecked())
		filteredClasses.push_back(15);
	if (linkerInsulatorCheckBox->isChecked())
		filteredClasses.push_back(16);
	if (personsCheckBox->isChecked())
		filteredClasses.push_back(17);
	if (trafficSignCheckBox->isChecked())
		filteredClasses.push_back(18);
	if (trafficLightCheckBox->isChecked())
		filteredClasses.push_back(19);
}

void ecvDeepSemanticSegmentationTool::selectAllClasses()
{
	bool state = selectAllRadioButton->isChecked();
	QList<QCheckBox*> list = classGroupBox->findChildren<QCheckBox*>();
	
	foreach(QCheckBox* ncheckBox, list)
	{
		if (ncheckBox)
		{
			ncheckBox->setChecked(state);
		}
	}
}

int ecvDeepSemanticSegmentationTool::performSegmentation()
{
	m_clusters_map.clear();
	m_clusters.clear();
	updateSelectedEntity();

	//check if selected entities are good
	int check_result = checkSelected();
	if (check_result != 1)
	{
		return check_result;
	}

	//if so go ahead with start()
	int start_status = startDetection();
	if (start_status != 1)
	{
		return start_status;
	}

	// verify segmentation result clusters
	size_t entityNumber = m_selectedEntity.getChildrenNumber();
	if (m_clusters_map.size() != entityNumber)
	{
		CVLog::Error("dimensions do not match!");
		return -1;
	}

	return 1;
}

int ecvDeepSemanticSegmentationTool::checkSelected()
{
	//In most of the cases we need at least 1 CC_POINT_CLOUD
	if (m_selectedEntity.getChildrenNumber() == 0)
	{
		CVLog::Warning("no selected point cloud, please select one again!");
		return -1;
	}

	std::vector<size_t> selectedFilters;
	getSelectedFilterClasses(selectedFilters);
	if (selectedFilters.empty())
	{
		CVLog::Warning("no selected class, please select at least one to continue!");
		return -1;
	}

	return 1;
}

static int s_computeStatus = 0;
static bool s_computing = false;
int ecvDeepSemanticSegmentationTool::startDetection()
{
	CVTools::TimeStart();

	if (s_computing)
	{
		return -1;
	}

	QString tipInfo = "Operation in progress, please wait for a while";
	if (useVotesCheckBox->isChecked())
	{
		tipInfo += "\n(disable votes to speed up!)";
	}

	QProgressDialog progressCb(tipInfo, QString(), 0, 0);

	if (m_show_progress)
	{
		progressCb.setWindowTitle(tr("Deep Semantic Segmentation"));
		progressCb.show();
		QApplication::processEvents();
	}

	s_computeStatus = -1;
	s_computing = true;
	int progress = 0;

	QFuture<void> future = QtConcurrent::run(this, &ecvDeepSemanticSegmentationTool::doCompute);
	while (!future.isFinished())
	{
#if defined(CV_WINDOWS)
		::Sleep(500);
#else
		usleep(500 * 1000);
#endif
		if (m_show_progress)
			progressCb.setValue(++progress);
	}

	int is_ok = s_computeStatus;
	s_computing = false;

	if (m_show_progress)
	{
		progressCb.close();
		QApplication::processEvents();
	}

	CVLog::Print(QString("Deep Semantic Segmentation: finish cost %1 s").arg(CVTools::TimeOff()));

	if (is_ok < 0)
	{
		return is_ok;
	}

	return 1;
}

void ecvDeepSemanticSegmentationTool::doCompute()
{
#ifdef ECV_PYTHON_USE_AS_DLL
	try
	{
		cloudViewer::utility::DeepSemanticSegmentation dss;
		dss.setEnableSampling(samplingCheckBox->isChecked());
		dss.setEnableVotes(useVotesCheckBox->isChecked());
		dss.setInputCloud(&m_selectedEntity);
		dss.compute(m_clusters, m_clusters_map);

		if (m_clusters_map.empty())
		{
			s_computeStatus = 1;
			return;
		}

		// filter to be segmented
		std::vector<size_t> filteredClasses;
		getSelectedFilterClasses(filteredClasses);
		for (auto itMap = m_clusters_map.begin(); itMap != m_clusters_map.end();)
		{
			auto &clusterMap = *itMap;
			for (auto it = clusterMap.begin(); it != clusterMap.end();)
			{
				int index = ClassMap::FindindexByValue(it->first);
				if (index < 0)
				{
					CVLog::Warning(QString("ignore class index {%1}").arg(index));
					++it;
					continue;
				}

				auto ret = std::find(filteredClasses.begin(), filteredClasses.end(), index);
				if (ret == filteredClasses.end() || it->second.size() < clusterMinSizeSpinBox->value())
				{
					it = clusterMap.erase(it);
					continue;
				}

				++it;
			}
			
			if (clusterMap.empty())
			{
				itMap = m_clusters_map.erase(itMap);
			}
			else
			{
				++itMap;
			}
		}
		
		s_computeStatus = m_clusters_map.empty() ? -1 : 1;
	}
	catch (const std::exception&)
	{
		s_computeStatus = -1;
	}

#else
	CVLog::Warning("python interface library has not been compiled!");
	return;
#endif // ECV_PYTHON_USE_AS_DLL
}

#endif
