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

#include "ecvFilterWindowTool.h"

// LOCAL
#include "ecvContourExtractor.h"
#include "ecvCropTool.h"
#include "MainWindow.h"
#include "ecvConsole.h"

// ECV_CORE_LIB
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>

// ECV_DB_LIB
#include <ecvPolyline.h>
#include <ecv2DViewportObject.h>

// LOCAL
#include "ecvBoundingBoxEditorDlg.h"
#include "ecvClippingBoxRepeatDlg.h"

using namespace DBLib;

//Last contour unique ID
static std::vector<unsigned> s_lastContourUniqueIDs;

//Contour extraction parameters
static double s_maxEdgeLength = -1.0;
static bool s_splitContours = false;
static bool s_multiPass = false;
static double s_defaultGap = 0.0;

// persistent map of the previous box used for each entity
struct ccClipBoxParams
{
	ccBBox box;
	ccGLMatrix trans;
};
static QMap< unsigned, ccClipBoxParams > s_lastBoxParams;

ecvFilterWindowTool::ecvFilterWindowTool(QMainWindow* parent)
	: ccOverlayDialog(parent)
	, Ui::GraphicalFilteringWindowDlg()
	, m_win(parent)
	, m_entityContainer("entities")
	, m_filtered(nullptr)
	, m_deleteHiddenParts(false)
	, m_somethingHasChanged(false)
	, m_currentMode(VTK_WIDGETS_TYPE::VTK_CLIP_WIDGET)
{
	setupUi(this);

	connect(resetButton, SIGNAL(clicked()), this, SLOT(reset()));
	connect(cancelButton, SIGNAL(clicked()), this, SLOT(cancel()));

	connect(exportButton, SIGNAL(clicked()), this, SLOT(exportSlice()));
	connect(exportMultButton, SIGNAL(clicked()), this, SLOT(exportMultSlices()));
	connect(extractContourToolButton, SIGNAL(clicked()), this, SLOT(extractContour()));
	connect(removeLastContourToolButton, SIGNAL(clicked()), this, SLOT(removeLastContour()));
	connect(restoreToolButton, SIGNAL(clicked()), this, SLOT(restoreLastBox()));

	QMenu* selectionModeMenu = new QMenu(this);
	selectionModeMenu->addAction(actionSegmentationExtraction);
	selectionModeMenu->addAction(actionPolylineExtraction);
	selectionModelButton->setDefaultAction(actionSegmentationExtraction);
	selectionModelButton->setMenu(selectionModeMenu);

	//selection modes
	connect(actionSegmentationExtraction, SIGNAL(triggered()), this, SLOT(doSetClipMode()));
	connect(actionPolylineExtraction, SIGNAL(triggered()), this, SLOT(doSetPolylineMode()));

	//add shortcuts
	addOverridenShortcut(Qt::Key_R); //return key for the "reset" button
	addOverridenShortcut(Qt::Key_Escape); //escape key for the "cancel" button
	addOverridenShortcut(Qt::Key_Tab);    //tab key to switch between rectangular and polygonal selection modes
	connect(this, SIGNAL(shortcutTriggered(int)), this, SLOT(onShortcutTriggered(int)));
}

ecvFilterWindowTool::~ecvFilterWindowTool()
{
	releaseAssociatedEntities();
}

void ecvFilterWindowTool::doSetClipMode()
{
	m_currentMode = VTK_WIDGETS_TYPE::VTK_CLIP_WIDGET;
	exportButton->setEnabled(true);
	exportMultButton->setEnabled(true);
	extractContourToolButton->setEnabled(false);
	removeLastContourToolButton->setEnabled(false);
	selectionModelButton->setDefaultAction(actionSegmentationExtraction);
	doSetClippingSelection();
}

void ecvFilterWindowTool::doSetPolylineMode()
{
	m_currentMode = VTK_WIDGETS_TYPE::VTK_SLICE_WIDGET;
	extractContourToolButton->setEnabled(true);
	removeLastContourToolButton->setEnabled(true);
	exportButton->setEnabled(false);
	exportMultButton->setEnabled(false);
	selectionModelButton->setDefaultAction(actionPolylineExtraction);
	doSetContourSelection();
}

void ecvFilterWindowTool::onShortcutTriggered(int key)
{
	switch (key)
	{

	case Qt::Key_Escape:
		cancelButton->click();
		return;	
	case Qt::Key_R:
		resetButton->click();
		return;

	case Qt::Key_Tab:
		if (m_currentMode == VTK_WIDGETS_TYPE::VTK_CLIP_WIDGET)
			doSetPolylineMode();
		else
			doSetClipMode();
		return;

	default:
		//nothing to do
		break;
	}
}

bool ecvFilterWindowTool::linkWith(QWidget* win)
{
	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}
	return true;
}

bool ecvFilterWindowTool::start()
{
	m_somethingHasChanged = false;
	reset();
	doSetClipMode();
	return ccOverlayDialog::start();
}

void ecvFilterWindowTool::doSetClippingSelection()
{
	m_currentMode = VTK_WIDGETS_TYPE::VTK_CLIP_WIDGET;
	linkWidgets();
}

void ecvFilterWindowTool::doSetContourSelection()
{
	m_currentMode = VTK_WIDGETS_TYPE::VTK_SLICE_WIDGET;
	linkWidgets();
}

void ecvFilterWindowTool::cancel()
{
	stop(true);
}

bool ecvFilterWindowTool::addAssociatedEntity(ccHObject* entity)
{
	if (!entity)
	{
		assert(false);
		return false;
	}
	
	//special case
	if (entity->isGroup())
	{
		for (unsigned i = 0; i < entity->getChildrenNumber(); ++i)
		{
			if (!addAssociatedEntity(entity->getChild(i)))
			{
				return false;
			}
		}
		return true;
	}


	bool firstEntity = (m_entityContainer.getChildrenNumber() == 0);
	if (firstEntity)
	{
		restoreToolButton->setEnabled(false);
	}

	m_entityContainer.addChild(entity, ccHObject::DP_NONE); //no dependency!

	if (s_lastBoxParams.contains(entity->getUniqueID()))
	{
		restoreToolButton->setEnabled(true);
	}

	if (entity->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		//contourGroupBox->setEnabled(true);
	}

	//no need to reset the clipping box if the entity has not a valid bounding-box
	if (!entity->getBB_recursive().isValid())
	{
		reset();
	}

	//force visibility
	entity->setVisible(true);
	entity->setEnabled(true);
	s_maxEdgeLength = -1.0;
	s_lastContourUniqueIDs.resize(0);
	return true;
}

unsigned ecvFilterWindowTool::getNumberOfAssociatedEntity() const
{
	return m_entityContainer.getChildrenNumber();
}

bool ecvFilterWindowTool::linkWidgets()
{
	QWidget* widget = ecvWidgetsInterface::LoadWidget(m_currentMode);
	if (!widget || !m_win)
	{
		return false;
	}
	ecvWidgetsInterface::SetInput(getOutput(), m_currentMode);
	MainWindow::TheInstance()->addWidgetToQMdiArea(widget);
	ecvDisplayTools::SetCurrentScreen(widget);
	return true;
}

void ecvFilterWindowTool::stop(bool accepted)
{
	reset();
	MainWindow::TheInstance()->addWidgetToQMdiArea(ecvDisplayTools::GetMainScreen());
	ecvDisplayTools::SetCurrentScreen(ecvDisplayTools::GetMainScreen());
	releaseAssociatedEntities();
	ccOverlayDialog::stop(accepted);
}

void ecvFilterWindowTool::releaseAssociatedEntities()
{
	for (unsigned ci = 0; ci < m_entityContainer.getChildrenNumber(); ++ci)
	{
		m_entityContainer.getChild(ci)->removeAllClipPlanes();
	}
	m_entityContainer.removeAllChildren();

	if (m_filtered)
	{
		m_filtered->removeAllChildren();
	}
}

void ecvFilterWindowTool::removeLastContour()
{
	if (s_lastContourUniqueIDs.empty())
		return;

	MainWindow* mainWindow = MainWindow::TheInstance();
	if (mainWindow)
	{
		for (size_t i = 0; i < s_lastContourUniqueIDs.size(); ++i)
		{
            ccHObject* obj = mainWindow->db()->find(static_cast<int>(s_lastContourUniqueIDs[i]));
			if (obj)
			{
				//obj->prepareDisplayForRefresh();
				mainWindow->removeFromDB(obj);
				ecvDisplayTools::RedrawDisplay();
			}
		}
	}

	s_lastContourUniqueIDs.resize(0);
}

ccHObject* ecvFilterWindowTool::getSlice(ccHObject* obj, bool silent)
{
	assert(m_box.isValid());
	if (!obj)
	{
		assert(false);
        return nullptr;
	}

	if (obj->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		ccGenericPointCloud* inputCloud = ccHObjectCaster::ToGenericPointCloud(obj);

		ccGenericPointCloud::VisibilityTableType selectionTable;
		try
		{
			selectionTable.resize(inputCloud->size());
		}
		catch (const std::bad_alloc&)
		{
			if (!silent)
			{
				CVLog::Error("Not enough memory!");
			}
            return nullptr;
		}
		flagPointsInside(inputCloud, &selectionTable);
		
		ccGenericPointCloud* sliceCloud = inputCloud->createNewCloudFromVisibilitySelection(false, &selectionTable, true);
		if (!sliceCloud)
		{
			if (!silent)
				CVLog::Error("Not enough memory!");
		}
		else if (sliceCloud->size() == 0)
		{
			delete sliceCloud;
			sliceCloud = nullptr;
		}
		return sliceCloud;
	}
	else if (obj->isKindOf(CV_TYPES::MESH))
	{
		ccGLMatrix transformation;

		const ccBBox& cropBox = m_box;
		ccHObject* mesh = ccCropTool::Crop(obj, cropBox, true, &transformation);
		if (!mesh)
		{
			if (!silent)
				CVLog::Error("Failed to segment the mesh!");
            return nullptr;
		}
		return mesh;
	}

    return nullptr;
}

void ecvFilterWindowTool::flagPointsInside(
	ccGenericPointCloud* cloud,
	ccGenericPointCloud::VisibilityTableType* visTable,
	bool shrink/*=false*/) const
{
	if (!cloud || !visTable)
	{
		//invalid input
		assert(false);
		return;
	}
	if (cloud->size() != visTable->size())
	{
		///size mismatch
		assert(false);
		return;
	}

	int count = static_cast<int>(cloud->size());

#if defined(_OPENMP)
#pragma omp parallel for
#endif
	for (int i = 0; i < count; ++i)
	{
		if (!shrink || visTable->at(i) == POINT_VISIBLE)
		{
			const CCVector3* P = cloud->getPoint(static_cast<unsigned>(i));
			visTable->at(i) = (m_box.contains(*P) ? POINT_VISIBLE : POINT_HIDDEN);
		}
	}
}

void ecvFilterWindowTool::exportSlice()
{
	if (!updateBBox())
	{
		CVLog::Warning("No available data can be exported!!!");
		return;
	}

	for (unsigned ci = 0; ci != m_entityContainer.getChildrenNumber(); ++ci)
	{
		ccHObject* obj = m_entityContainer.getChild(ci);
		if (!obj)
		{
			assert(false);
			continue;
		}
		ccHObject* result = getSlice(obj, false);

		if (result)
		{
			result->setName(obj->getName() + QString(".section"));
			if (obj->getParent())
				obj->getParent()->addChild(result);
			MainWindow::TheInstance()->addToDB(result);
		}
	}

	resetButton->setEnabled(true);
}

bool ecvFilterWindowTool::updateBBox()
{
	m_filtered = ecvWidgetsInterface::GetOutput(m_currentMode);
	if (!m_filtered)
	{
		return false;
	}

	if (m_filtered->getChildrenNumber())
	{
		m_box = m_filtered->getBB_recursive();
	}
	
	return true;
}

void ecvFilterWindowTool::exportMultSlices()
{
	extractSlicesAndContours(true, true, /*singleContourMode=*/false);
}

void ecvFilterWindowTool::extractContour()
{
	extractSlicesAndContours(false, true, /*singleContourMode=*/true);
}

static unsigned ComputeGridDimensions(	const ccBBox& localBox,
										const bool processDim[3],
										int indexMins[3],
										int indexMaxs[3],
										int gridDim[3],
										const CCVector3& gridOrigin,
										const CCVector3& cellSizePlusGap)
{
	//compute 'grid' extents in the local clipping box ref.
	for (int i=0; i<3; ++i)
	{
		indexMins[i] = 0;
		indexMaxs[i] = 0;
		gridDim[i]   = 1;
	}
	unsigned cellCount = 1;

	for (unsigned char d = 0; d < 3; ++d)
	{
		if (processDim[d])
		{
            if (CVLib::LessThanEpsilon(cellSizePlusGap.u[d]))
			{
				CVLog::Error("Box size (plus gap) is null! Can't apply repetitive process!");
				return 0;
			}

			// don't forget the user defined gap between 'cells'
			PointCoordinateType a = (localBox.minCorner().u[d] - gridOrigin.u[d]) / cellSizePlusGap.u[d]; 
			PointCoordinateType b = (localBox.maxCorner().u[d] - gridOrigin.u[d]) / cellSizePlusGap.u[d];

			indexMins[d] = static_cast<int>(floor(a + static_cast<PointCoordinateType>(1.0e-6)));
			indexMaxs[d] = static_cast<int>(ceil(b - static_cast<PointCoordinateType>(1.0e-6))) - 1;

			assert(indexMaxs[d] >= indexMins[d]);
			gridDim[d] = std::max(indexMaxs[d] - indexMins[d] + 1, 1);
			cellCount *= static_cast<unsigned>(gridDim[d]);
		}
	}

	return cellCount;
}

ccBBox ecvFilterWindowTool::getOwnBB() const
{
	return m_box;
}

bool ecvFilterWindowTool::extractFlatContour(
	ccPointCloud* sliceCloud,
	bool allowMultiPass,
	PointCoordinateType maxEdgeLength,
	std::vector<ccPolyline*>& parts,
	bool allowSplitting/*=true*/,
	const PointCoordinateType* preferredDim/*=0*/,
	bool enableVisualDebugMode/*=false*/
)
{
	assert(sliceCloud);
	parts.clear();
	{
		//create vertices
		ccPointCloud* contourVertices = sliceCloud->cloneThis();
		contourVertices->setName("vertices");
		contourVertices->setEnabled(false);

		assert(contourVertices);
		int hullPtsCount = contourVertices->size();
		ccPolyline* basePoly = new ccPolyline(contourVertices);
		if (basePoly->reserve(hullPtsCount))
		{
			basePoly->addPointIndex(0, hullPtsCount);
			basePoly->setClosed(false);
			basePoly->setVisible(true);
			basePoly->setName("contour");
			basePoly->addChild(contourVertices);
		}

		//extract whole contour
		if (!basePoly)
		{
			return false;
		}
		else if (!allowSplitting)
		{
			parts.push_back(basePoly);
			return true;
		}

		//and split it if necessary
		bool success = basePoly->split(maxEdgeLength, parts);
		delete basePoly;
		basePoly = 0;
		return success;
	}
}

bool ecvFilterWindowTool::extractSlicesAndContours
(
	const std::vector<ccGenericPointCloud*>& clouds,
	const std::vector<ccGenericMesh*>& meshes,
	ccBBox& clipBox,
	bool singleContourMode,
	bool repeatDimensions[3],
	std::vector<ccHObject*>& outputSlices,
	bool extractContours,
	PointCoordinateType maxEdgeLength,
	std::vector<ccPolyline*>& outputContours,
	PointCoordinateType gap/*=0*/,
	bool multiPass/*=false*/,
	bool splitContours/*=false*/,
	bool projectOnBestFitPlane/*=false*/,
	bool visualDebugMode/*=false*/,
	bool generateRandomColors/*=false*/,
	ecvProgressDialog* progressDialog/*=0*/)
{
	//check input
	if (clouds.empty() && meshes.empty())
	{
		assert(false);
		return false;
	}

	//repeat dimensions
	int repeatDimensionsSum = static_cast<int>(repeatDimensions[0])
							+ static_cast<int>(repeatDimensions[1])
							+ static_cast<int>(repeatDimensions[2]);

	if (!singleContourMode && repeatDimensionsSum == 0)
	{
		assert(false);
		CVLog::Error("No dimension selected to repeat the segmentation process?!");
		return false;
	}

	//compute the cloud bounding box in the local clipping box ref.
	ccGLMatrix localTrans;

	CCVector3 gridOrigin = getOwnBB().minCorner();
	CCVector3 cellSize = getOwnBB().getDiagVec();
	CCVector3 cellSizePlusGap = cellSize + CCVector3(gap, gap, gap);

	//apply process
	try
	{
		bool error = false;
		bool warningsIssued = false;
		size_t cloudSliceCount = 0;

		if (singleContourMode)
		{
			//single contour: easy
			outputSlices.reserve(clouds.size());
			for (size_t ci = 0; ci != clouds.size(); ++ci)
			{
				ccHObject* slice = getSlice(clouds[ci], false);
				if (slice)
				{
					slice->setName(clouds[ci]->getName() + QString(".slice"));
					outputSlices.push_back(slice);
				}
			}

			if (outputSlices.empty() && !extractContours)
			{
				//error message already issued
				return false;
			}
			cloudSliceCount = outputSlices.size();
		}
		else //repeat mode
		{
			if (!clouds.empty()) //extract sections from clouds
			{
				//compute 'grid' extents in the local clipping box ref.
				ccBBox localBox;
				for (ccGenericPointCloud* cloud : clouds)
				{
					for (unsigned i = 0; i < cloud->size(); ++i)
					{
						CCVector3 P = *cloud->getPoint(i);
						localTrans.apply(P);
						localBox.add(P);
					}
				}

				int indexMins[3], indexMaxs[3], gridDim[3];
				unsigned cellCount = ComputeGridDimensions(localBox, repeatDimensions, indexMins, indexMaxs, gridDim, gridOrigin, cellSizePlusGap);

				//we'll potentially create up to one (ref.) cloud per input loud and per cell
				std::vector<CVLib::ReferenceCloud*> refClouds;
				refClouds.resize(cellCount * clouds.size(), 0);

				if (progressDialog)
				{
					progressDialog->setWindowTitle(tr("Preparing extraction"));
					progressDialog->start();
					progressDialog->show();
					progressDialog->setAutoClose(false);
				}

				unsigned subCloudsCount = 0;

				//project points into grid
				for (size_t ci = 0; ci != clouds.size(); ++ci)
				{
					ccGenericPointCloud* cloud = clouds[ci];
					unsigned pointCount = cloud->size();

					QString infos = tr("Cloud '%1").arg(cloud->getName());
					infos += tr("Points: %L1").arg( pointCount );
					if (progressDialog)
					{
						progressDialog->setInfo(infos);
					}
					QApplication::processEvents();

					CVLib::NormalizedProgress nProgress(progressDialog, pointCount);
					for (unsigned i = 0; i < pointCount; ++i)
					{
						CCVector3 P = *cloud->getPoint(i);
						localTrans.apply(P);

						//relative coordinates (between 0 and 1)
						P -= gridOrigin;
						P.x /= cellSizePlusGap.x;
						P.y /= cellSizePlusGap.y;
						P.z /= cellSizePlusGap.z;

						int xi = static_cast<int>(floor(P.x));
						xi = std::min(std::max(xi, indexMins[0]), indexMaxs[0]);
						int yi = static_cast<int>(floor(P.y));
						yi = std::min(std::max(yi, indexMins[1]), indexMaxs[1]);
						int zi = static_cast<int>(floor(P.z));
						zi = std::min(std::max(zi, indexMins[2]), indexMaxs[2]);

						if (gap == 0 ||
							(	(P.x - static_cast<PointCoordinateType>(xi))*cellSizePlusGap.x <= cellSize.x
							&&	(P.y - static_cast<PointCoordinateType>(yi))*cellSizePlusGap.y <= cellSize.y
							&&	(P.z - static_cast<PointCoordinateType>(zi))*cellSizePlusGap.z <= cellSize.z))
						{
							int cloudIndex = ((zi - indexMins[2]) * static_cast<int>(gridDim[1]) + (yi - indexMins[1])) * static_cast<int>(gridDim[0]) + (xi - indexMins[0]);
							assert(cloudIndex >= 0 && static_cast<size_t>(cloudIndex)* clouds.size() + ci < refClouds.size());

							CVLib::ReferenceCloud*& destCloud = refClouds[cloudIndex * clouds.size() + ci];
							if (!destCloud)
							{
								destCloud = new CVLib::ReferenceCloud(cloud);
								++subCloudsCount;
							}

							if (!destCloud->addPointIndex(i))
							{
								CVLog::Error("Not enough memory!");
								error = true;
								break;
							}
						}
					}

					nProgress.oneStep();
				} //project points into grid

				if (progressDialog)
				{
					progressDialog->setWindowTitle(QObject::tr("Section extraction"));
					progressDialog->setInfo(QObject::tr("Section(s): %L1").arg(subCloudsCount));
					progressDialog->setMaximum(static_cast<int>(subCloudsCount));
					progressDialog->setValue(0);
					QApplication::processEvents();
				}

				//reset count
				subCloudsCount = 0;

				//now create the real clouds
				for (int i = indexMins[0]; i <= indexMaxs[0]; ++i)
				{
					for (int j = indexMins[1]; j <= indexMaxs[1]; ++j)
					{
						for (int k = indexMins[2]; k <= indexMaxs[2]; ++k)
						{
							int cloudIndex = ((k - indexMins[2]) * static_cast<int>(gridDim[1]) + (j - indexMins[1])) * static_cast<int>(gridDim[0]) + (i - indexMins[0]);
							assert(cloudIndex >= 0 && static_cast<size_t>(cloudIndex)* clouds.size() < refClouds.size());

							for (size_t ci = 0; ci != clouds.size(); ++ci)
							{
								ccGenericPointCloud* cloud = clouds[ci];
								CVLib::ReferenceCloud* destCloud = refClouds[cloudIndex * clouds.size() + ci];
								if (destCloud) //some slices can be empty!
								{
									//generate slice from previous selection
									int warnings = 0;
									ccPointCloud* sliceCloud = cloud->isA(CV_TYPES::POINT_CLOUD) ? static_cast<ccPointCloud*>(cloud)->partialClone(destCloud, &warnings) : ccPointCloud::From(destCloud, cloud);
									warningsIssued |= (warnings != 0);

									if (sliceCloud)
									{
										if (generateRandomColors)
										{
											ecvColor::Rgb col = ecvColor::Generator::Random();
											if (!sliceCloud->setRGBColor(col))
											{
												CVLog::Error("Not enough memory!");
												error = true;
												i = indexMaxs[0];
												j = indexMaxs[1];
												k = indexMaxs[2];
											}
											sliceCloud->showColors(true);
										}

										sliceCloud->setEnabled(true);
										sliceCloud->setVisible(true);
										//sliceCloud->setDisplay(cloud->getDisplay());

										CCVector3 cellOrigin(	gridOrigin.x + i * cellSizePlusGap.x,
																gridOrigin.y + j * cellSizePlusGap.y,
																gridOrigin.z + k * cellSizePlusGap.z);
										QString slicePosStr = QString("(%1 ; %2 ; %3)").arg(cellOrigin.x).arg(cellOrigin.y).arg(cellOrigin.z);
										sliceCloud->setName(QString("slice @ ") + slicePosStr);

										//add slice to group
										outputSlices.push_back(sliceCloud);
										++subCloudsCount;

										if (progressDialog)
										{
											progressDialog->setValue(static_cast<int>(subCloudsCount));
										}
									}

									if (progressDialog && progressDialog->wasCanceled())
									{
										error = true;
										CVLog::Warning(QString("[ExtractSlicesAndContours] Process canceled by user"));
										//early stop
										i = indexMaxs[0];
										j = indexMaxs[1];
										k = indexMaxs[2];
										break;
									}
								}
							}
						}
					}
				} //now create the real clouds

				//release memory
				{
					for (size_t i = 0; i < refClouds.size(); ++i)
						if (refClouds[i])
							delete refClouds[i];
					refClouds.clear();
				}

				cloudSliceCount = outputSlices.size();

			} //extract sections from clouds

			if (!meshes.empty()) //extract sections from meshes
			{
				//compute 'grid' extents in the local clipping box ref.
				ccBBox localBox;
				for (ccGenericMesh* mesh : meshes)
				{
					ccGenericPointCloud* cloud = mesh->getAssociatedCloud();
					for (unsigned i = 0; i < cloud->size(); ++i)
					{
						CCVector3 P = *cloud->getPoint(i);
						localTrans.apply(P);
						localBox.add(P);
					}
				}

				int indexMins[3], indexMaxs[3], gridDim[3];
				unsigned cellCount = ComputeGridDimensions(localBox, repeatDimensions, indexMins, indexMaxs, gridDim, gridOrigin, cellSizePlusGap);

				const ccGLMatrix* _transformation = 0;
				ccGLMatrix transformation;
				if (progressDialog)
				{
					progressDialog->setWindowTitle("Section extraction");
					progressDialog->setInfo(QObject::tr("Up to (%1 x %2 x %3) = %4 section(s)").arg(gridDim[0]).arg(gridDim[1]).arg(gridDim[2]).arg(cellCount));
					progressDialog->setMaximum(static_cast<int>(cellCount * meshes.size()));
					progressDialog->show();
					QApplication::processEvents();
				}

				//now extract the slices
				for (int i = indexMins[0]; i <= indexMaxs[0]; ++i)
				{
					for (int j = indexMins[1]; j <= indexMaxs[1]; ++j)
					{
						for (int k = indexMins[2]; k <= indexMaxs[2]; ++k)
						{
							int sliceIndex = ((k - indexMins[2]) * static_cast<int>(gridDim[1]) + (j - indexMins[1])) * static_cast<int>(gridDim[0]) + (i - indexMins[0]);

							CCVector3 C = gridOrigin + CCVector3(i*cellSizePlusGap.x, j*cellSizePlusGap.y, k*cellSizePlusGap.z);
							ccBBox cropBox(C, C + cellSize);

							for (size_t mi = 0; mi != meshes.size(); ++mi)
							{
								ccGenericMesh* mesh = meshes[mi];
								ccHObject* croppedEnt = ccCropTool::Crop(mesh, cropBox, true, _transformation);
								if (croppedEnt)
								{
									if (generateRandomColors)
									{
										ccPointCloud* croppedVertices = ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
										if (croppedVertices)
										{
											ecvColor::Rgb col = ecvColor::Generator::Random();
											if (!croppedVertices->setRGBColor(col))
											{
												CVLog::Error("Not enough memory!");
												error = true;
												i = indexMaxs[0];
												j = indexMaxs[1];
												k = indexMaxs[2];
											}
											croppedVertices->showColors(true);
											mesh->showColors(true);
										}
									}

									croppedEnt->setEnabled(true);
									croppedEnt->setVisible(true);
									//croppedEnt->setDisplay(mesh->getDisplay());

									QString slicePosStr = QString("(%1 ; %2 ; %3)").arg(C.x).arg(C.y).arg(C.z);
									croppedEnt->setName(QString("slice @ ") + slicePosStr);

									//add slice to group
									outputSlices.push_back(croppedEnt);
								}

								if (progressDialog)
								{
									if (progressDialog->wasCanceled())
									{
										error = true;
										CVLog::Warning(QString("[ExtractSlicesAndContours] Process canceled by user"));
										//early stop
										i = indexMaxs[0];
										j = indexMaxs[1];
										k = indexMaxs[2];
										break;
									}
									progressDialog->setValue(sliceIndex * static_cast<int>(meshes.size()) + static_cast<int>(mi));
								}
							}
						}
					}
				}
			} //extract sections from meshes

		} //repeat mode

		//extract contour polylines (optionaly)
		if (!error && extractContours)
		{
			assert(m_filtered);
			unsigned filteredNum = m_filtered->getChildrenNumber();

			if (progressDialog)
			{
				progressDialog->setWindowTitle("Contour extraction");
				progressDialog->setInfo(QObject::tr("Contour(s): %L1").arg(filteredNum));
				progressDialog->setMaximum(static_cast<int>(filteredNum));
				if (!visualDebugMode)
				{
					progressDialog->show();
					QApplication::processEvents();
				}
			}

			//preferred dimension?
			int preferredDim = -1;
			if (repeatDimensionsSum == 1 && !projectOnBestFitPlane)
			{
				for (int i = 0; i < 3; ++i)
					if (repeatDimensions[i])
						preferredDim = i;
			}

			ccGLMatrix invLocalTrans = localTrans.inverse();
			PointCoordinateType* preferredOrientation = (preferredDim != -1 ? invLocalTrans.getColumn(preferredDim) : 0);

			//we create the corresponding (3D) polyline
			// process all the slices originating from point clouds
			for (unsigned i = 0; i != filteredNum; ++i)
			{
				ccHObject* obj = m_filtered->getChild(i);
				ccPointCloud* sliceCloud = ccHObjectCaster::ToPointCloud(obj);
				assert(sliceCloud);

				std::vector<ccPolyline*> polys;
				if (extractFlatContour(sliceCloud,
					multiPass,
					maxEdgeLength,
					polys,
					splitContours,
					preferredOrientation,
					visualDebugMode))
				{
					if (!polys.empty())
					{
						for (size_t p = 0; p < polys.size(); ++p)
						{
							ccPolyline* poly = polys[p];
							poly->setColor(ecvColor::green);
							poly->showColors(true);
							poly->set2DMode(false);
							poly->setGlobalScale(sliceCloud->getGlobalScale());
							poly->setGlobalShift(sliceCloud->getGlobalShift());
							QString contourName = sliceCloud->getName();
							contourName.replace("slice", "contour");
							if (polys.size() > 1)
							{
								contourName += QString(" (part %1)").arg(p + 1);
							}
							poly->setName(contourName);
							outputContours.push_back(poly);
						}
					}
					else
					{
						CVLog::Warning(QString("%1: points are too far from each other! Increase the max edge length").arg(sliceCloud->getName()));
						warningsIssued = true;
					}
				}
				else
				{
					CVLog::Warning(QString("contour extraction failed!"));
					warningsIssued = true;
				}

				if (progressDialog && !visualDebugMode)
				{
					if (progressDialog->wasCanceled())
					{
						error = true;
						CVLog::Warning(QString("[ExtractSlicesAndContours] Process canceled by user"));
						//early stop
						break;
					}
					progressDialog->setValue(static_cast<int>(i));
				}
			}

		} //extract contour polylines

		//release memory
		if (error || singleContourMode)
		{
			for (ccHObject* slice : outputSlices)
			{
				delete slice;
			}
			outputSlices.resize(0);
		}

		if (error)
		{
			for (ccPolyline* poly : outputContours)
			{
				delete poly;
			}
			return false;
		}
		else if (warningsIssued)
		{
			CVLog::Warning("[ExtractSlicesAndContours] Warnings were issued during the process! (result may be incomplete)");
		}
	}
	catch (const std::bad_alloc&)
	{
		CVLog::Error("Not enough memory!");
		return false;
	}

	return true;
}

void ecvFilterWindowTool::extractSlicesAndContours(bool extractSlices, bool extractContours, bool singleContourMode)
{
	if (m_entityContainer.getChildrenNumber() == 0)
	{
		assert(false);
		return;
	}

	if (!updateBBox())
	{
		CVLog::Warning("No available data can be exported!!!");
		return;
	}

	std::vector<ccGenericPointCloud*> clouds;
	std::vector<ccGenericMesh*> meshes;
	try
	{
		for (unsigned ci = 0; ci != m_entityContainer.getChildrenNumber(); ++ci)
		{
			ccHObject* obj = m_entityContainer.getChild(ci);
			if (obj->isKindOf(CV_TYPES::POINT_CLOUD))
			{
				ccGenericPointCloud* cloud = ccHObjectCaster::ToGenericPointCloud(obj);
				clouds.push_back(cloud);
			}
			else if (obj->isKindOf(CV_TYPES::MESH))
			{
				ccGenericMesh* mesh = ccHObjectCaster::ToGenericMesh(obj);
				meshes.push_back(mesh);
			}
		}
	}
	catch (const std::bad_alloc&)
	{
		CVLog::Warning("Not enough memory");
		return;
	}

	if (clouds.empty() && meshes.empty())
	{
		CVLog::Warning("Only works with point clouds or meshes!");
		return;
	}

	ccClippingBoxRepeatDlg repeatDlg(singleContourMode, MainWindow::TheInstance());
	repeatDlg.extractContoursGroupBox->setEnabled(true);
	
	//by default we set the 'flat/repeat' dimension to the smallest box dimension
	{
		CCVector3 diagVec = getOwnBB().getDiagVec();
		unsigned char flatDim = 0;
		if (diagVec.y < diagVec.x)
			flatDim = 1;
		if (diagVec.z < diagVec.u[flatDim])
			flatDim = 2;
		if (singleContourMode)
			repeatDlg.setFlatDim(flatDim);
		else
			repeatDlg.setRepeatDim(flatDim);
	}
	
	//random colors is only useful for mutliple slice/contour mode
	repeatDlg.randomColorCheckBox->setEnabled(!singleContourMode);
	
	//set default max edge length
	if (s_maxEdgeLength < 0)
		s_maxEdgeLength = getOwnBB().getDiagNorm();
	repeatDlg.maxEdgeLengthDoubleSpinBox->setValue(s_maxEdgeLength);
	repeatDlg.splitContourCheckBox->setChecked(s_splitContours);
	repeatDlg.multiPassCheckBox->setChecked(s_multiPass);
	repeatDlg.gapDoubleSpinBox->setValue(s_defaultGap);

	if (!repeatDlg.exec())
	{
		//cancelled by user
		return;
	}

	//repeat dimensions 
	bool processDim[3] = {	repeatDlg.xRepeatCheckBox->isChecked(),
							repeatDlg.yRepeatCheckBox->isChecked(),
							repeatDlg.zRepeatCheckBox->isChecked() };

	//whether to use random colors for (multiple) generated slices
	s_defaultGap = repeatDlg.gapDoubleSpinBox->value();
	s_maxEdgeLength = repeatDlg.maxEdgeLengthDoubleSpinBox->value();
	s_splitContours = repeatDlg.splitContourCheckBox->isChecked();
	s_multiPass = repeatDlg.multiPassCheckBox->isChecked();
	s_defaultGap = repeatDlg.gapDoubleSpinBox->value();
	bool projectOnBestFitPlane = repeatDlg.projectOnBestFitCheckBox->isChecked();
	bool visualDebugMode = repeatDlg.debugModeCheckBox->isChecked();
	bool generateRandomColors = repeatDlg.randomColorCheckBox->isChecked();

	//whether to extract contours or not
	if (!singleContourMode)
	{
		extractContours = repeatDlg.extractContoursGroupBox->isChecked();
	}

	ecvProgressDialog pDlg(false, m_win);
	std::vector<ccHObject*> outputSlices;
	std::vector<ccPolyline*> outputContours;

	QElapsedTimer eTimer;
	eTimer.start();

	if (!extractSlicesAndContours(clouds,
									meshes,
									m_box,
									singleContourMode,
									processDim,
									outputSlices,
									extractContours,
									static_cast<PointCoordinateType>(s_maxEdgeLength),
									outputContours,
									static_cast<PointCoordinateType>(s_defaultGap),
									s_multiPass,
									s_splitContours,
									projectOnBestFitPlane,
									visualDebugMode,
									generateRandomColors,
									&pDlg
									))
	{
		//process failed (error message has already been issued)
		return;
	}

	CVLog::Print("[ecvFilterWindowTool] Processed finished in %.2f s.", eTimer.elapsed() / 1.0e3);

	//base name
	QString baseName;
	if (m_entityContainer.getChildrenNumber() == 1)
	{
		baseName = m_entityContainer.getFirstChild()->getName();
	}

	//slices
	if (!outputSlices.empty())
	{
		QString groupName;
		if (!baseName.isEmpty())
		{
			groupName = QString("%1.slices").arg(baseName);
		}
		else
		{
			groupName = "Slices";
		}
		ccHObject* sliceGroup = new ccHObject(groupName);

		for (ccHObject* slice : outputSlices)
		{
			sliceGroup->addChild(slice);
		}

		QMessageBox::warning(0, "Process finished", 
			QString("%1 slices have been generated.\n(you may have to close the tool and hide the initial cloud to see them...)").
			arg(sliceGroup->getChildrenNumber()));
		MainWindow::TheInstance()->addToDB(sliceGroup);
	}
	else if (!singleContourMode)
	{
		CVLog::Warning("[ecvFilterWindowTool] Repeat process generated no output!");
	}

	//contour polylines
	if (!outputContours.empty())
	{
		ccHObject* contourGroup = new ccHObject(baseName.isEmpty() ? QString("Contours") : baseName + QString(".contours"));

		for (ccPolyline* poly : outputContours)
		{
			contourGroup->addChild(poly);
		}
		MainWindow::TheInstance()->addToDB(contourGroup);

		s_lastContourUniqueIDs.clear();
		s_lastContourUniqueIDs.push_back(contourGroup->getUniqueID());
	}

	//ecvDisplayTools::RedrawDisplay();
}

void ecvFilterWindowTool::reset()
{
	m_box.clear();
	if (m_entityContainer.getChildrenNumber())
	{
		m_box = m_entityContainer.getBB_recursive();
	}

	if (m_somethingHasChanged)
	{
		m_somethingHasChanged = false;
	}

	if (m_filtered)
	{
		m_filtered->removeAllChildren();
		delete m_filtered;
		m_filtered = nullptr;
	}

	resetButton->setEnabled(false);
}

void ecvFilterWindowTool::restoreLastBox()
{
	if (m_entityContainer.getChildrenNumber() == 0)
	{
		assert(false);
		return;
	}
	
	unsigned uniqueID = m_entityContainer.getFirstChild()->getUniqueID();
	if (!s_lastBoxParams.contains(uniqueID))
	{
		assert(false);
		return;
	}

	const ccClipBoxParams& params = s_lastBoxParams[uniqueID];
	m_box = params.box;
}
