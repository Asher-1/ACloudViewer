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

#include "ecvRenderSurfaceTool.h"

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

ecvRenderSurfaceTool::ecvRenderSurfaceTool(QMainWindow* parent)
	: ccOverlayDialog(parent)
	, Ui::GraphicalRenderSurfaceWindowDlg()
	, m_win(parent)
	, m_entityContainer("entities")
	, m_filtered(nullptr)
	, m_deleteHiddenParts(false)
	, m_somethingHasChanged(false)
	, m_currentMode(VTK_WIDGETS_TYPE::VTK_SURFACE_WIDGET)
{
	setupUi(this);

	connect(resetButton, SIGNAL(clicked()), this, SLOT(reset()));
	connect(cancelButton, SIGNAL(clicked()), this, SLOT(cancel()));

	connect(exportMeshSurfaceButton, SIGNAL(clicked()), this, SLOT(exportSurface()));
	connect(exportPointCloudButton, SIGNAL(clicked()), this, SLOT(exportPointCloud()));
	connect(restoreToolButton, SIGNAL(clicked()), this, SLOT(restoreLastBox()));

	//add shortcuts
	addOverridenShortcut(Qt::Key_R); //return key for the "reset" button
	addOverridenShortcut(Qt::Key_Escape); //escape key for the "cancel" button
	addOverridenShortcut(Qt::Key_Tab);    //tab key to switch between rectangular and polygonal selection modes
	connect(this, SIGNAL(shortcutTriggered(int)), this, SLOT(onShortcutTriggered(int)));
}

ecvRenderSurfaceTool::~ecvRenderSurfaceTool()
{
	releaseAssociatedEntities();
}

void ecvRenderSurfaceTool::onShortcutTriggered(int key)
{
	switch (key)
	{

	case Qt::Key_Escape:
		cancelButton->click();
		return;	
	case Qt::Key_R:
		resetButton->click();
		return;
	default:
		//nothing to do
		break;
	}
}

bool ecvRenderSurfaceTool::linkWith(QWidget* win)
{
	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}
	return true;
}

bool ecvRenderSurfaceTool::start()
{
	m_somethingHasChanged = false;
	reset();
	linkWidgets();
	return ccOverlayDialog::start();
}

void ecvRenderSurfaceTool::doSetClippingSelection()
{
	m_currentMode = VTK_WIDGETS_TYPE::VTK_SURFACE_WIDGET;
	linkWidgets();
}

void ecvRenderSurfaceTool::cancel()
{
	stop(true);
}

bool ecvRenderSurfaceTool::addAssociatedEntity(ccHObject* entity)
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

unsigned ecvRenderSurfaceTool::getNumberOfAssociatedEntity() const
{
	return m_entityContainer.getChildrenNumber();
}

bool ecvRenderSurfaceTool::linkWidgets()
{
	QWidget* widget = ecvWidgetsInterface::LoadWidget(m_currentMode);
	if (!widget || !m_win)
	{
		return false;
	}
	ecvWidgetsInterface::SetInput(getOutput(), m_currentMode);
	ecvDisplayTools::SetCurrentScreen(widget);
	return true;
}

void ecvRenderSurfaceTool::stop(bool accepted)
{
	reset();
	ecvDisplayTools::SetCurrentScreen(ecvDisplayTools::GetMainScreen());
	releaseAssociatedEntities();
	ccOverlayDialog::stop(accepted);
}

void ecvRenderSurfaceTool::releaseAssociatedEntities()
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

void ecvRenderSurfaceTool::exportSurface()
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
		ccHObject* result = 0;

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

bool ecvRenderSurfaceTool::updateBBox()
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

void ecvRenderSurfaceTool::exportPointCloud()
{
}

ccBBox ecvRenderSurfaceTool::getOwnBB() const
{
	return m_box;
}

void ecvRenderSurfaceTool::reset()
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

void ecvRenderSurfaceTool::restoreLastBox()
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