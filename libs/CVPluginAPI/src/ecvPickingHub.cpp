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
//#                    COPYRIGHT: CLOUDVIEWER  project                     #
//#                                                                        #
//##########################################################################

#include "ecvPickingHub.h"

// ECV_DB_LIB
#include <ecvDisplayTools.h>

// Qt
#include <QMdiSubWindow>

// Plugins
#include <ecvMainAppInterface.h>

ccPickingHub::ccPickingHub(ecvMainAppInterface* app, QObject* parent/*=0*/)
	: QObject(parent)
	, m_app(app)
	, m_activeWindow(nullptr)
	, m_pickingMode(ecvDisplayTools::POINT_OR_TRIANGLE_PICKING)
	, m_autoEnableOnActivatedWindow(true)
	, m_exclusive(false)
{
}

void ccPickingHub::togglePickingMode(bool state)
{
	//CVLog::Warning(QString("Toggle picking mode: ") + (state ? "ON" : "OFF") + " --> " + (m_activeGLWindow ? QString("View ") + QString::number(m_activeGLWindow->getUniqueID()) : QString("no view")));
	if (m_activeWindow)
	{
		ecvDisplayTools::SetPickingMode(state ? 
			m_pickingMode : ecvDisplayTools::DEFAULT_PICKING);
	}
}

void ccPickingHub::onActiveWindowChanged(QMdiSubWindow* mdiSubWindow)
{
	QWidget* window = (mdiSubWindow ? mdiSubWindow->widget() : nullptr);
	//if (glWindow)
    //	CVLog::Warning("New active GL window: " + glWindow->getViewId());
	//else
	//	CVLog::Warning("No more active GL window");

	if (m_activeWindow == window)
	{
		//nothing to do
		return;
	}

	if (m_activeWindow)
	{
		//take care of the previously linked window
		togglePickingMode(false);
		disconnect(m_activeWindow);
		m_activeWindow = nullptr;
	}

	if (window)
	{
		//link this new window
		connect(ecvDisplayTools::TheInstance(), &ecvDisplayTools::itemPicked, 
			this, &ccPickingHub::processPickedItem, Qt::UniqueConnection);
		connect(ecvDisplayTools::TheInstance(), &QObject::destroyed, 
			this, &ccPickingHub::onActiveWindowDeleted);
		m_activeWindow = window;

		if (m_autoEnableOnActivatedWindow && !m_listeners.empty())
		{
			togglePickingMode(true);
		}
	}
}

void ccPickingHub::onActiveWindowDeleted(QObject* obj)
{
	if (obj == m_activeWindow)
	{
		m_activeWindow = nullptr;
	}
}

void ccPickingHub::processPickedItem(ccHObject* entity, unsigned itemIndex, int x, int y, const CCVector3& P3D)
{
	if (m_listeners.empty())
	{
		return;
	}

	ccPickingListener::PickedItem item;
	{
		item.clickPoint = QPoint(x, y);
		item.entity = entity;
		item.itemIndex = itemIndex;
		item.P3D = P3D;
	}

	//copy the list of listeners, in case the user call 'removeListener' in 'onItemPicked'
	std::set< ccPickingListener* > listeners = m_listeners;

	for (ccPickingListener* l : listeners)
	{
		if (l)
		{
			l->onItemPicked(item);
		}
	}
}

bool ccPickingHub::addListener(	ccPickingListener* listener,
								bool exclusive/*=false*/,
								bool autoStartPicking/*=true*/,
								ecvDisplayTools::PICKING_MODE mode/*=ccGLWindow::POINT_OR_TRIANGLE_PICKING*/)
{
	if (!listener)
	{
		assert(false);
		return false;
	}

	//if listeners are already registered
	if (!m_listeners.empty())
	{
		if (m_exclusive) //a previous listener is exclusive
		{
			assert(m_listeners.size() == 1);
			if (m_listeners.find(listener) == m_listeners.end())
			{
				CVLog::Warning("[ccPickingHub::addListener] Exclusive listener already registered: stop the other tool relying on point picking first");
				return false;
			}
		}
		else if (exclusive) //this new listener is exclusive
		{
			if (m_listeners.size() > 1 || m_listeners.find(listener) == m_listeners.end())
			{
				CVLog::Warning("[ccPickingHub::addListener] Attempt to register an exclusive listener while other listeners are already registered");
				return false;
			}
		}
		else if (mode != m_pickingMode)
		{
			if (m_listeners.size() > 1 || m_listeners.find(listener) == m_listeners.end())
			{
				CVLog::Warning("[ccPickingHub::addListener] Other listeners are already registered with a different picking mode");
				return false;
			}
		}
	}
	
	try
	{
		m_listeners.insert(listener);
	}
	catch (const std::bad_alloc&)
	{
		//not enough memory
		CVLog::Warning("[ccPickingHub::addListener] Not enough memory");
		return false;
	}

	m_exclusive = exclusive;
	m_pickingMode = mode;

	if (autoStartPicking)
	{
		togglePickingMode(true);
	}

	return true;
}

void ccPickingHub::removeListener(ccPickingListener* listener, bool autoStopPickingIfLast/*=true*/)
{
	m_listeners.erase(listener);

	if (m_listeners.empty())
	{
		m_exclusive = false; //auto drop the 'exclusive' flag
		togglePickingMode(false);
	}
}
