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

#include "ecvFilterTool.h"

// LOCAL
#include "ecvClippingBoxRepeatDlg.h"
#include "MainWindow.h"

// ECV_CORE_LIB
#include <ecvGenericFiltersTool.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>

// ECV_DB_LIB
#include <ecvPolyline.h>

// QT
#include <QMessageBox>

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

ecvFilterTool::ecvFilterTool(QWidget* parent)
	: ccOverlayDialog(parent)
	, Ui::FilterToolDlg()
	, m_filter(nullptr)
{
	setupUi(this);

	connect(exportButton,					SIGNAL(clicked()),				this, SLOT(exportSlice()));
	connect(resetButton,					SIGNAL(clicked()),				this, SLOT(reset()));
	connect(restoreToolButton,				SIGNAL(clicked()),				this, SLOT(restoreOrigin()));
	connect(closeButton,					SIGNAL(clicked()),				this, SLOT(closeDialog()));
	
	connect(showBoxToolButton,				SIGNAL(toggled(bool)),			this, SLOT(toggleBox(bool)));
	connect(showInteractorsToolButton,		SIGNAL(toggled(bool)),			this, SLOT(toggleInteractors(bool)));

	connect(minusXShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftXMinus()));
	connect(plusXShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftXPlus()));
	connect(minusYShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftYMinus()));
	connect(plusYShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftYPlus()));
	connect(minusZShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftZMinus()));
	connect(plusZShiftToolButton,			SIGNAL(clicked()),				this, SLOT(shiftZPlus()));

	viewButtonsFrame->setEnabled(true);
	connect(viewUpToolButton,				SIGNAL(clicked()),				this, SLOT(setTopView()));
	connect(viewDownToolButton,				SIGNAL(clicked()),				this, SLOT(setBottomView()));
	connect(viewFrontToolButton,			SIGNAL(clicked()),				this, SLOT(setFrontView()));
	connect(viewBackToolButton,				SIGNAL(clicked()),				this, SLOT(setBackView()));
	connect(viewLeftToolButton,				SIGNAL(clicked()),				this, SLOT(setLeftView()));
	connect(viewRightToolButton,			SIGNAL(clicked()),				this, SLOT(setRightView()));

	s_maxEdgeLength = -1.0;
}

ecvFilterTool::~ecvFilterTool()
{
}

void ecvFilterTool::toggleInteractors(bool state)
{
	m_filter->showInteractor(state);
}

void ecvFilterTool::toggleBox(bool state)
{
	if (m_filter)
		m_filter->showOutline(state);
}

bool ecvFilterTool::addAssociatedEntity(ccHObject* entity)
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

	if (!m_filter)
	{
		CVLog::Error(QString("[ecvFilterTool::addAssociatedEntity] No associated filter module!"));
		return false;
	}

	if (!m_entityContainer.addChild(entity, ccHObject::DP_NONE))
	{
		//error message already issued
		CVLog::Error("An error occurred (see Console)");
		return false;
	}

	//force visibility
	entity->setVisible(true);
	entity->setEnabled(true);
	return true;
}

unsigned ecvFilterTool::getNumberOfAssociatedEntity() const
{
	return m_entityContainer.getChildrenNumber();
}

bool ecvFilterTool::linkWith(QWidget* win)
{
	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}
	
	return true;
}

bool ecvFilterTool::start()
{
	assert(!m_processing);
	if (!m_filter)
		return false;

	m_filter->setInputData(m_entityContainer.getFirstChild());
	m_filter->start();
	m_filter->showOutline(false);
	m_out_entities.clear();

	return ccOverlayDialog::start();
}

void ecvFilterTool::stop(bool state)
{
	if (m_filter)
	{
		if (state && getNumberOfAssociatedEntity())
		{
			//save clip box parameters
			ccClipBoxParams params;
			m_filter->get(params.box, params.trans);
			for (unsigned ci = 0; ci != getNumberOfAssociatedEntity(); ++ci)
			{
				s_lastBoxParams[m_entityContainer.getChild(ci)->getUniqueID()] = params;
			}
		}

		parametersLayout->removeWidget(m_filter->getFilterWidget());

		m_filter->unregisterFilter();
		if (m_filter)
		{
			delete m_filter;
			m_filter = nullptr;
		}

		releaseAssociatedEntities();
	}

	ccOverlayDialog::stop(state);
}

void ecvFilterTool::setFilter(ecvGenericFiltersTool * filter)
{
	if (!filter) return;
	m_filter = filter;
	m_filterType = filter->getFilterType();
	parametersLayout->addWidget(filter->getFilterWidget());
}

ccHObject* ecvFilterTool::getSlice(bool silent)
{
	if (!m_filter) return nullptr;

	ccHObject* obj = m_filter->getOutput();
	if (!obj)
	{
		if (!silent)
			CVLog::Error("No output or Not enough memory!");
	}

	return obj;
}

void ecvFilterTool::exportSlice()
{
	if (!m_filter || !MainWindow::TheInstance())
		return;

	for (unsigned ci = 0; ci != getNumberOfAssociatedEntity(); ++ci)
	{
		ccHObject* obj = m_entityContainer.getChild(ci);
		if (!obj)
		{
			assert(false);
			continue;
		}

		if (m_filterType == ecvGenericFiltersTool::PROBE_FILTER)
		{
			// save plot figure internally other than entity
			ccHObject* result = getSlice(true);
		}
		else if (m_filterType == ecvGenericFiltersTool::GLYPH_FILTER)
		{
			// do nothing
			CVLog::Warning("Glyph filter has no output, just for showing!");
			return;
		}
		else
		{
			ccHObject* result = getSlice(false);
			if (result)
			{
				m_out_entities.push_back(result);
				result->setEnabled(false);
				result->setName(obj->getName() + QString(".section"));
				if (obj->getParent())
					obj->getParent()->addChild(result);
				MainWindow::TheInstance()->addToDB(result);
			}
			else
			{
				CVLog::Warning("failed to get filter result");
			}
		}
	}

	// reset interactors or model if necessary
	reset();
}

void ecvFilterTool::releaseAssociatedEntities()
{
	m_entityContainer.removeAllChildren();
}

void ecvFilterTool::shiftBox(unsigned char dim, bool minus)
{
	if (!m_filter) return;
	const ccBBox& bbox = m_filter->getBox();
	if (!bbox.isValid()) return;

	assert(dim<3);

	PointCoordinateType width = (bbox.maxCorner() - bbox.minCorner()).u[dim];
	CCVector3 shiftVec(0, 0, 0);
	shiftVec.u[dim] = (minus ? -width : width);
	m_filter->shift(shiftVec);
}

void ecvFilterTool::reset()
{
	if (m_filter)
		m_filter->reset();
}

void ecvFilterTool::restoreOrigin()
{
	if (m_filter)
	{
		m_filter->restore();
	}
}

void ecvFilterTool::closeDialog()
{
	stop(true);
}

ccBBox ecvFilterTool::getSelectedEntityBbox()
{
	ccBBox box;
	if (getNumberOfAssociatedEntity() != 0)
	{
		box = m_entityContainer.getDisplayBB_recursive(false);
	}
	return box;
}

void ecvFilterTool::setView(CC_VIEW_ORIENTATION orientation)
{
	ccBBox* bbox = nullptr;
	ccBBox box = getSelectedEntityBbox();
	if (box.isValid())
	{
		bbox = &box;
	}
	ecvDisplayTools::SetView(orientation, bbox);
}
