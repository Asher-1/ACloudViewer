#include "widgetsSurfaceInterface.h"
#include "dialog.h"
#include "contourwindow.h"
#include "rendersurface.h"

// ECV_DB_LIB
#include <ecvHObject.h>

using namespace DBLib;

QString SurfaceWidgetInterface::getInterfaceName() const
{
	return QObject::tr("Surface Interface");
}

QStringList SurfaceWidgetInterface::getWidgtsNames() const
{
	QStringList widgetsName;
	widgetsName << "RenderSurface";
	return widgetsName;
}

QList<QWidget*> SurfaceWidgetInterface::getWidgts() const
{
	if (m_widgetsMap.isEmpty())
	{
		return QList<QWidget*>();
	}
	return m_widgetsMap.values();
}


QWidget * SurfaceWidgetInterface::getWidgtByID(const VTK_WIDGETS_TYPE widgetID) const
{
	if (!m_widgetsMap.contains(widgetID))
	{
		return nullptr;
	}
	return m_widgetsMap[widgetID];
}

const QString SurfaceWidgetInterface::getWidgtNameByID(const VTK_WIDGETS_TYPE widgetID) const
{
	if (!m_NamesMap.contains(widgetID))
	{
		return QString();
	}
	return m_NamesMap[widgetID];
}

bool SurfaceWidgetInterface::setInput(const ccHObject * obj, const VTK_WIDGETS_TYPE widgetType)
{
	if (!obj || m_widgetsMap.isEmpty() || m_NamesMap.isEmpty())
	{
		return false;
	}

	QWidget * widget = getWidgtByID(widgetType);
	if (!widget)
	{
		return false;
	}

	RenderSurface * surface = static_cast<RenderSurface *>(widget);
	if (!surface)
	{
		return false;
	}

	return surface->setInput(obj);
}

ccHObject * SurfaceWidgetInterface::getOutput(const DBLib::VTK_WIDGETS_TYPE widgetType)
{
	if (m_widgetsMap.isEmpty() || m_NamesMap.isEmpty())
	{
		return nullptr;
	}

	RenderSurface * surface = static_cast<RenderSurface *>(getWidgtByID(widgetType));
	if (!surface)
	{
		return nullptr;
	}

	return surface->getOutput();
}

void SurfaceWidgetInterface::initWidgets()
{
	if (!m_widgetsMap.isEmpty() || !m_NamesMap.isEmpty())
	{
		return;
	}

	QStringList widgetNames = getWidgtsNames();

	for (QString & name : widgetNames)
	{
		if (name == QString("RenderSurface"))
		{
			m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_SURFACE_WIDGET, new RenderSurface());
			m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_SURFACE_WIDGET, name);
		}
		else if (name == QString("None"))
		{
			// TODO
		}
	}
}

void SurfaceWidgetInterface::unregister()
{
	if (m_widgetsMap.isEmpty())
	{
		return;
	}

	QMap<VTK_WIDGETS_TYPE, QWidget*>::iterator iter = m_widgetsMap.begin();
	while (iter != m_widgetsMap.end())
	{
		QWidget * widget = iter.value();
		if (widget)
		{
			delete widget;
			widget = nullptr;
		}
		iter++;
	}

	m_widgetsMap.clear();
	m_NamesMap.clear();
}
