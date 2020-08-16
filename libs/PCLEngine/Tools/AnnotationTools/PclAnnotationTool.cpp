//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
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
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#include "PclAnnotationTool.h" 

//Local
#include "PclUtils/cc2sm.h"
#include "PclUtils/PCLVis.h"
#include "PclUtils/PCLConv.h"
#include "Tools/ecvTools.h"
#include "Tools/AnnotationTools/Annotaion.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVGeom.h>
#include <CVTools.h>
#include <ClassMap.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

// VTK
#include <vtkRenderWindowInteractor.h>

// QT
#include <QDir>
#include <QFileInfo>

#define DEFAULT_POINT 0
#define SELECTED_POINT -2
#define GROUND_POINT -1

using namespace PclUtils;

PclAnnotationTool::PclAnnotationTool(AnnotationMode mode)
	: ecvGenericAnnotationTool(mode)
{
}

PclAnnotationTool::PclAnnotationTool(
	ecvGenericVisualizer3D* viewer, AnnotationMode mode)
	: ecvGenericAnnotationTool(mode)
{
	this->initialize(viewer);
}

PclAnnotationTool::~PclAnnotationTool()
{
}

////////////////////Initialization///////////////////////////
void PclAnnotationTool::initialize(ecvGenericVisualizer3D* viewer)
{
	assert(viewer);
	setVisualizer(viewer);

	resetMode();

	//init annotation manager
	m_annoManager.reset(new Annotaions(
		m_annotationMode == AnnotationMode::BOUNDINGBOX ?
		m_viewer->getRenderWindowInteractor() : nullptr));
	
	m_currPickedAnnotation = nullptr;

	// init m_baseCloud
	m_baseCloud.reset(new PointCloudI);
	m_cloudLabel = nullptr;
}

bool PclAnnotationTool::loadClassesFromFile(const std::string & file)
{
	std::ifstream input(file.c_str(), std::ios_base::in);
	if(!input.good()){
		CVLog::Error(tr("Cannot open file : %1").arg(file.c_str()));
		return false;
	}

	std::vector<std::string> labels;
	std::string value;
	while (input >> value) {
		if (value == "") continue;
		labels.push_back(value);
	}
	CVLog::Print(tr("load %1 classes from %2").arg(labels.size()).arg(file.c_str()));
	initAnnotationLabels(labels);
	return true;
}

void PclAnnotationTool::initAnnotationLabels(const std::vector<std::string>& labelList)
{
	if (labelList.empty()) return;

	// cache annotation type history
	std::vector<int> tempIndex;
	for (auto anno : m_annoManager->getAnnotations()) {
		tempIndex.push_back(Annotation::GetTypeIndex(anno->getType()));
	}

	// init label type
	Annotation::GetTypes()->clear();
	for (size_t i = 0; i < labelList.size(); ++i)
	{
		if (labelList[i] == "")
		{
			continue;
		}
		Annotation::GetTypes()->push_back(labelList[i]);
	}

	// update existing annotation type
	int index = -1;

	vector<Annotation *> toBeRemovedAnnotations;
	for (auto anno : m_annoManager->getAnnotations()) {
		index++;
		if (Annotation::GetTypes()->size() - 1 < tempIndex[index])
		{
			toBeRemovedAnnotations.push_back(anno);
			CVLog::Warning(tr("drop annotaion[%1] which is out of range or current classSets").arg(anno->getType().c_str()));
			continue;
		}

		anno->setType(Annotation::GetTypes()->at(tempIndex[index]));
		m_annoManager->updateBalloonByIndex(index);
		labelCloudByAnnotation(anno);
	}

	// remove invalid annotations
	for (size_t i = 0; i < toBeRemovedAnnotations.size(); ++i)
	{
		removeAnnotation(toBeRemovedAnnotations[i]);
	}

	updateCloud();
}

void PclAnnotationTool::toggleInteractor()
{
	if (!m_viewer) return;

	m_viewer->toggleAreaPicking();
}

void PclAnnotationTool::getAnnotationLabels(std::vector<std::string>& labelList)
{
	// get label types
	labelList.clear();
	for (size_t i = 0; i < Annotation::GetTypes()->size(); ++i)
	{
		if (Annotation::GetTypes()->at(i) == "")
		{
			continue;
		}
		labelList.push_back(Annotation::GetTypes()->at(i));
	}
}

bool PclAnnotationTool::getCurrentAnnotations(std::vector<int>& annos) const
{
	return m_annoManager && m_annoManager->getAnnotations(annos);
}

bool PclAnnotationTool::setInputCloud(ccPointCloud* cloud, int viewPort)
{
	PCLCloud::Ptr smCloud = cc2smReader(cloud).getAsSM();
	if (!smCloud)
	{
		return false;
	}

	FROM_PCL_CLOUD(*smCloud, *this->m_baseCloud);
	if (this->m_baseCloud->size() < 1)
	{
		return false;
	}

	// hide origin cloud
	{
		m_baseCloudId = QString::number(cloud->getUniqueID()).toStdString();
		vtkActor* modelActor = m_viewer->getActorById(m_baseCloudId);
		if (modelActor)
		{
			modelActor->SetVisibility(0);
		}
	}

	m_annoManager->preserve(this->m_baseCloud->size());

	m_pointcloudFileName = CVTools::fromQString(cloud->getFullPath());
	QFileInfo fileInfo(cloud->getFullPath());

	QString annoName = fileInfo.baseName();
	if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
	{
		annoName = fileInfo.baseName() + ".boxes";
	}
	else if (m_annotationMode == AnnotationMode::SEMANTICS)
	{
		annoName = fileInfo.baseName() + ".labels";
	}
	
	QDir dir(fileInfo.absolutePath());

	// 1. load classes file if exists in curren file path
	QString classesFile = dir.absoluteFilePath("Classets.classes");
	if (!(QFile::exists(classesFile) && loadClassesFromFile(CVTools::fromQString(classesFile))))
	{
		// load default classes!
		this->loadDefaultClasses();
	}

	// 2. load labels or boxes annotaion file if exists in curren file path
	m_annotationFileName = CVTools::fromQString(dir.absoluteFilePath(annoName));
	{
		if (QFile::exists(CVTools::toQString(m_annotationFileName))) {
			m_annoManager->loadAnnotations(m_annotationFileName, m_annotationMode);
			if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
			{
				CVLog::Print(tr("%1: load %2 boxes").
					arg(m_annotationFileName.c_str()).arg(m_annoManager->getSize()));
			}
			else if (m_annotationMode == AnnotationMode::SEMANTICS)
			{
				CVLog::Print(tr("%1: load %2 classes").
					arg(m_annotationFileName.c_str()).arg(m_annoManager->getSize()));
			}
		}
	}

	refresh();
	return true;
}

void PclAnnotationTool::refresh()
{
	m_cloudLabel = new int[m_baseCloud->size()];
	memset(m_cloudLabel, DEFAULT_POINT, m_baseCloud->size() * sizeof(int));
	labelCloudByAnnotations();

	m_colorHandler.setInputCloud(m_baseCloud);
	m_colorHandler.setLabel(m_cloudLabel);
	m_viewer->addPointCloud<PointIntensity>(m_baseCloud, m_colorHandler, m_annotationCloudId, 0);

	// show annotation if exists
	showAnnotation();
	updateCloud();
}

void PclAnnotationTool::start()
{
	if (m_viewer)
	{
		m_viewer->setAreaPickingEnabled(true);
		m_viewer->setPointPickingEnabled(false);

		if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
		{
			m_viewer->setActorPickingEnabled(true);
		}

		connect(m_viewer, &PCLVis::interactorPickedEvent, this, &PclAnnotationTool::pickedEventProcess);
		connect(m_viewer, &PCLVis::interactorKeyboardEvent, this, &PclAnnotationTool::keyboardEventProcess);
		connect(m_viewer, &PCLVis::interactorAreaPickedEvent, this, &PclAnnotationTool::areaPickingEventProcess);
	}
}

void PclAnnotationTool::stop()
{
	if (m_viewer)
	{
		m_viewer->setAreaPickingEnabled(false);
		m_viewer->setPointPickingEnabled(true);

		if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
		{
			m_viewer->setActorPickingEnabled(false);
		}

		disconnect(m_viewer, &PCLVis::interactorPickedEvent, this, &PclAnnotationTool::pickedEventProcess);
		disconnect(m_viewer, &PCLVis::interactorKeyboardEvent, this, &PclAnnotationTool::keyboardEventProcess);
		disconnect(m_viewer, &PCLVis::interactorAreaPickedEvent, this, &PclAnnotationTool::areaPickingEventProcess);
	}

	resetMode();
	clear();
	//ecvDisplayTools::SimulateKeyBoardPress(Qt::Key_Q, 3);
	CVLog::Print("unregister annotations tool successfully");
}

void PclAnnotationTool::setVisualizer(ecvGenericVisualizer3D* viewer)
{
	if (viewer)
	{
		m_viewer = reinterpret_cast<PCLVis*>(viewer);
		if (!m_viewer)
		{
			CVLog::Warning("[PclAnnotationTool::setVisualizer] viewer is Null!");
		}
	}
	else
	{
		CVLog::Warning("[PclAnnotationTool::setVisualizer] viewer is Null!");
	}
}
//////////////////////////////////////////////////////////////

////////////////////Register callback////////////////////////
void PclAnnotationTool::intersectMode()
{
	m_intersectMode = true;
	m_unionMode = false;
	m_trimMode = false;
}

void PclAnnotationTool::unionMode()
{
	m_intersectMode = false;
	m_unionMode = true;
	m_trimMode = false;
}

void PclAnnotationTool::trimMode()
{
	m_intersectMode = false;
	m_unionMode = false;
	m_trimMode = true;
}

void PclAnnotationTool::resetMode()
{
	m_intersectMode = false;
	m_unionMode = false;
	m_trimMode = false;
}

////////////////////Callback function///////////////////////////
void PclAnnotationTool::pointPickingProcess(int index)
{
}

void PclAnnotationTool::areaPickingEventProcess(const std::vector<int>& new_selected_slice)
{
	if (new_selected_slice.empty() || !m_viewer) return;

	int s = m_viewer->getRenderWindowInteractor()->GetShiftKey();
	int c = m_viewer->getRenderWindowInteractor()->GetControlKey();
	int a = m_viewer->getRenderWindowInteractor()->GetAltKey();
	bool skip = a;

	// remove existed annotated points
	vector<int> selected_slice;
	filterPickedSlice(new_selected_slice, selected_slice, skip);
	if (selected_slice.empty()) return;

	if (!m_last_selected_slice.empty()) {
		defaultColorPoint(m_last_selected_slice);
	}

	if (s && c || m_intersectMode) { // intersection
		m_last_selected_slice = ecvTools::IntersectionVector(m_last_selected_slice, selected_slice);
	}
	else if (s || m_unionMode) { // union
		m_last_selected_slice = ecvTools::UnionVector(m_last_selected_slice, selected_slice);
	}
	else if (c || m_trimMode) { // remove
		m_last_selected_slice = ecvTools::DiffVector(m_last_selected_slice, selected_slice);
	}
	else { // new
		m_last_selected_slice = selected_slice;
	}

	highlightPoint(m_last_selected_slice);
	updateCloud();
}

void PclAnnotationTool::pickedEventProcess(vtkActor* actor) 
{
	if (m_currPickedAnnotation) {
		m_currPickedAnnotation->unpicked();
		m_currPickedAnnotation = nullptr;
	}

	if (!actor)
	{
		return;
	}

	// get the correspond annotation
	m_currPickedAnnotation = m_annoManager->getAnnotation(actor);
	if (m_currPickedAnnotation) {
		m_currPickedAnnotation->picked(m_viewer->getRenderWindowInteractor());
		emit ecvGenericAnnotationTool::objectPicked(true);
	}
	else
	{
		emit ecvGenericAnnotationTool::objectPicked(false);
	}
}

void PclAnnotationTool::keyboardEventProcess(const std::string& symKey)
{
	// delete annotation
	if (symKey == "Delete") {

		if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
		{
			if (m_currPickedAnnotation)
			{
				CVLog::Print(tr("remove box annotation [%1] with %2 points")
					.arg(m_currPickedAnnotation->getType().c_str())
					.arg(m_currPickedAnnotation->getSlice().size()));
				removeAnnotation(m_currPickedAnnotation);
				updateCloud();
			}
		}
		else if (m_annotationMode == AnnotationMode::SEMANTICS)
		{
			if (!m_annoManager->getAnnotations().empty())
			{
				Annotation* anno = m_annoManager->getAnnotations().back();
				CVLog::Print(tr("remove last annotation [%1] with %2 points")
					.arg(anno->getType().c_str())
					.arg(anno->getSlice().size()));
				removeAnnotation(anno);
				updateCloud();
			}
			else
			{
				CVLog::Warning("no annotation exists!");
			}
		}
	}
}
///////////////////////////////////////////////////////////////


////////////////////Core processing////////////////////////////
void PclAnnotationTool::filterPickedSlice(
	const std::vector<int>& inSlices, 
	std::vector<int>& outSlices, bool skip)
{
	if (!outSlices.empty())
	{
		outSlices.clear();
	}

	for (auto &x : inSlices) {
		if (m_annoManager->getLabelByIndex(x) == DEFAULT_POINT || skip) {
			outSlices.push_back(x);
		}
	}
}

void PclAnnotationTool::fastLabelCloud(const std::vector<int>& inSlices, int label)
{
	int num = static_cast<int>(inSlices.size());
	if (m_cloudLabel && num != 0)
	{
#ifdef USE_TBB
		tbb::parallel_for(0, num, [&](int dataIndex)
#else

#if defined(_OPENMP)
#pragma omp parallel for
#endif
		for (int dataIndex = 0; dataIndex < num; ++dataIndex)
#endif
		{
			m_cloudLabel[inSlices[dataIndex]] = label;
		}
#ifdef USE_TBB
		);
#endif
	}
}

void PclAnnotationTool::changeAnnotationType(const std::string& type)
{
	// create annotation from last selected slice
	if (m_last_selected_slice.size() > 3) {
		createAnnotationFromSelectPoints(type);
	}
	else
	{
		// show annotation interactor if current picked annotation is not None
		if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
		{
			if (m_currPickedAnnotation) {
				CVLog::Print(tr("change current picked annotation type from [%1] to [%2].").
					arg(m_currPickedAnnotation->getType().c_str(), type.c_str()));
				changeAnnotationType(m_currPickedAnnotation, type);
			}
			else
			{
				CVLog::Warning(tr("no box picked now! please pick one box and try again!"));
			}
		}
		else if (m_annotationMode == AnnotationMode::SEMANTICS)
		{
			vector<Annotation *>& annos = m_annoManager->getAnnotations();
			if (!annos.empty() &&
				annos.back()->getType() != type)
			{
				CVLog::Print(tr("change last annotation type from [%1] to [%2]").
					arg(annos.back()->getType().c_str(), type.c_str()));
				changeAnnotationType(annos.back(), type);
			}
			else
			{
				CVLog::Warning(tr("no annotation exists now! please create one and try again!"));
			}
		}
	}
}

void PclAnnotationTool::selectExistedAnnotation(const std::string & type)
{
	if (m_annotationMode == AnnotationMode::SEMANTICS)
	{

		m_lastSelectedAnnotations.clear();
		m_annoManager->getAnnotations(type, m_lastSelectedAnnotations);
		if (m_lastSelectedAnnotations.empty())
		{
			CVLog::Warning(tr("cannot find annotation type [%1], ignore it!").arg(type.c_str()));
			return;
		}
		else
		{
			m_last_selected_slice.clear();
			for (Annotation* anno : m_lastSelectedAnnotations)
			{
				if (anno)
				{
					m_last_selected_slice.insert(m_last_selected_slice.end(), anno->getSlice().begin(), anno->getSlice().end());
					m_annoManager->remove(anno);
				}
			}

			if (m_last_selected_slice.empty())
			{
				return;
			}

			highlightPoint(m_last_selected_slice);
			updateCloud();
		}

	}
}

void PclAnnotationTool::changeAnnotationType(Annotation* anno, const std::string & type)
{
	if (!anno) return;

	anno->setType(type);
	m_annoManager->updateBalloonByAnno(anno);
	showAnnotation(anno);
	updateCloud();
}

void PclAnnotationTool::exportAnnotations()
{
	m_annoManager->saveAnnotations(m_annotationFileName, int(m_annotationMode));
	CVLog::Print(tr("annotations file has been saved to %1").arg(CVTools::toQString(m_annotationFileName)));
}

void PclAnnotationTool::createAnnotationFromSelectPoints(std::string type)
{
	if (m_last_selected_slice.size() > 3) {
		Annotation* anno;
		if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
		{
			anno = new Annotation(m_baseCloud, m_last_selected_slice, type);
		}
		else if (m_annotationMode == AnnotationMode::SEMANTICS)
		{
			anno = new Annotation(m_last_selected_slice, type);
		}

		m_annoManager->add(anno);
		m_last_selected_slice.clear();
		showAnnotation(anno);
		updateCloud();
	}
	else {
		CVLog::Warning(tr("no points selected or selected points number is less than 3"));
	}
}

void PclAnnotationTool::labelCloudByAnnotations()
{
	if (!m_cloudLabel) return;
	for (auto anno : m_annoManager->getAnnotations()) {
		labelCloudByAnnotation(anno);
	}
}

void PclAnnotationTool::labelCloudByAnnotation(const Annotation* anno)
{
	if (!anno || anno->getSlice().empty()) return;
	fastLabelCloud(anno->getSlice(), Annotation::GetTypeIndex(anno->getType()));
}

void PclAnnotationTool::resetCloudByAnnotation(const Annotation * anno)
{
	if (!m_viewer || 
		!m_viewer->contains(m_annotationCloudId) ||
		!anno || anno->getSlice().empty())
	{
		return;
	}

	fastLabelCloud(anno->getSlice(), DEFAULT_POINT);
}

void PclAnnotationTool::updateCloudLabel(const std::string& type)
{
	if (!m_cloudLabel || m_last_selected_slice.size() < 1) return;

	fastLabelCloud(m_last_selected_slice, Annotation::GetTypeIndex(type));
}

void PclAnnotationTool::loadDefaultClasses()
{
	std::vector<std::string> labels;
	for (auto it : ClassMap::SemanticMap) {
		labels.push_back(it.second);
	}

	initAnnotationLabels(labels);
}
///////////////////////////////////////////////////////////////


////////////////////Visualization///////////////////////////
void PclAnnotationTool::highlightPoint(std::vector<int>& slice)
{
	if (slice.size() < 1) return;

	fastLabelCloud(slice, SELECTED_POINT);
}

void PclAnnotationTool::defaultColorPoint(std::vector<int>& slice)
{
	if (!m_viewer || !m_viewer->contains(m_annotationCloudId)) return;

	if (slice.empty()) {
		if (m_cloudLabel)
		{
			memset(m_cloudLabel, DEFAULT_POINT, m_baseCloud->size() * sizeof(int));
		}
		return;
	}

	fastLabelCloud(slice, DEFAULT_POINT);
}

void PclAnnotationTool::groundColorPoint(std::vector<int>& slice)
{
	if (slice.size() < 1) return;

	fastLabelCloud(slice, GROUND_POINT);
}

void PclAnnotationTool::updateCloud()
{
	if (!m_viewer || m_viewer->contains(m_annotationCloudId))
	{
		m_viewer->updatePointCloud<PointIntensity>(m_baseCloud, m_colorHandler, m_annotationCloudId);
		ecvDisplayTools::UpdateScreen();
	}
}

void PclAnnotationTool::setPointSize(const std::string & viewID, int viewPort)
{
	if (!m_viewer) return;
	m_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, viewID, viewPort);
}

void PclAnnotationTool::showAnnotation()
{
	for (auto anno : m_annoManager->getAnnotations()) {
		showAnnotation(anno);
	}
}

void PclAnnotationTool::hideAnnotation()
{
	for (auto anno : m_annoManager->getAnnotations()) {
		hideAnnotation(anno);
	}
}

void PclAnnotationTool::showOrigin()
{
	if (!m_viewer) return;

	vtkActor* originActor = m_viewer->getActorById(m_baseCloudId);
	if (originActor)
	{
		originActor->SetVisibility(1);
	}

	vtkActor* annotatedActor = m_viewer->getActorById(m_annotationCloudId);
	if (annotatedActor)
	{
		annotatedActor->SetVisibility(0);
	}
}

void PclAnnotationTool::hideOrigin()
{
	vtkActor* originActor = m_viewer->getActorById(m_baseCloudId);
	if (originActor)
	{
		originActor->SetVisibility(0);
	}

	vtkActor* annotatedActor = m_viewer->getActorById(m_annotationCloudId);
	if (annotatedActor)
	{
		annotatedActor->SetVisibility(1);
	}
}

void PclAnnotationTool::reset()
{
	removeAnnotation();
	m_currPickedAnnotation = nullptr;
	m_last_selected_slice.clear();
}

void PclAnnotationTool::clear()
{
	if (!m_viewer) return;

	m_viewer->removePointCloud(m_annotationCloudId);
	reset();
	m_annoManager->release();

	// show origin cloud
	{
		vtkActor* modelActor = m_viewer->getActorById(m_baseCloudId);
		if (modelActor)
		{
			modelActor->SetVisibility(1);
		}
	}

	if (m_cloudLabel) {
		delete[] m_cloudLabel;
		m_cloudLabel = nullptr;
	}
}

void PclAnnotationTool::showAnnotation(const Annotation* anno) {
	if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
	{
		if (!m_viewer) return;
		m_viewer->addActorToRenderer(anno->getActor());
	}
	labelCloudByAnnotation(anno);
}

void PclAnnotationTool::removeAnnotation()
{
	hideAnnotation();
	m_annoManager->clear();
	std::vector<int> temp;
	defaultColorPoint(temp);
	updateCloud();
}

void PclAnnotationTool::removeAnnotation(Annotation * anno)
{
	hideAnnotation(anno);
	m_annoManager->remove(anno);
}

void PclAnnotationTool::hideAnnotation(Annotation *anno)
{
	if (m_currPickedAnnotation) {
		m_currPickedAnnotation->unpicked();
		m_currPickedAnnotation = nullptr;
	}

	resetCloudByAnnotation(anno);

	if (m_annotationMode == AnnotationMode::BOUNDINGBOX)
	{
		if (!m_viewer) return;
		m_viewer->removeActorFromRenderer(anno->getActor());
	}
}
///////////////////////////////////////////////////////////////
