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

#include "ecvPointPickingGenericInterface.h"

//Local
#include "MainWindow.h"
#include "db_tree/ecvDBRoot.h"

// CV_CORE_LIB
#include <CVLog.h>

//common
#include <ecvPickingHub.h>

//ECV_DB_LIB
#include <ecvPointCloud.h>
#include <ecvDisplayTools.h>

ccPointPickingGenericInterface::ccPointPickingGenericInterface(ccPickingHub* pickingHub, QWidget* parent/*=0*/)
	: ccOverlayDialog(parent)
	, m_pickingHub(pickingHub)
{
	assert(m_pickingHub);
}

bool ccPointPickingGenericInterface::linkWith(QWidget* win)
{
	//just in case
	if (m_pickingHub)
	{
		m_pickingHub->removeListener(this);
	}

	if (!ccOverlayDialog::linkWith(win))
	{
		return false;
	}

	//if the dialog is already linked to a window, we must disconnect the 'point picked' signal
	return true;
}

bool ccPointPickingGenericInterface::start()
{
	if (!m_pickingHub)
	{
		CVLog::Error("[Point picking] No associated display!");
		return false;
	}

	//activate "point picking mode" in associated GL window
	if (!m_pickingHub->addListener(this, true, true, ecvDisplayTools::POINT_PICKING))
	{
		CVLog::Error("Picking mechanism already in use. Close the tool using it first.");
		return false;
	}

	ccOverlayDialog::start();
	return true;
}

void ccPointPickingGenericInterface::stop(bool state)
{
	if (m_pickingHub)
	{
		//deactivate "point picking mode" in all GL windows
		m_pickingHub->removeListener(this);
	}

	ccOverlayDialog::stop(state);
}

void ccPointPickingGenericInterface::onItemPicked(const PickedItem& pi)
{
	if (!m_processing || !pi.entity)
		return;

	if (pi.entity->isKindOf(CV_TYPES::POINT_CLOUD))
	{
		ccPointCloud* cloud = static_cast<ccPointCloud*>(pi.entity);
		if (!cloud)
		{
			assert(false);
			CVLog::Warning("[Item picking] Picked point is not in pickable entities DB?!");
			return;
		}
		processPickedPoint(cloud, pi.itemIndex, pi.clickPoint.x(), pi.clickPoint.y());
	}
	else if (pi.entity->isKindOf(CV_TYPES::MESH))
	{
		//NOT HANDLED: 'POINT_PICKING' mode only for now
		CVLog::Warning("[Item picking] 'POINT_PICKING' mode only for now!");
		return;
		//assert(false);
	}
	else
	{
		//unhandled entity
		assert(false);
	}
}
