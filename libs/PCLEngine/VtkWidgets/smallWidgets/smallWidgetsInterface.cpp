#include "smallWidgetsInterface.h"
#include "distancewidgetwindow.h"
#include "anglewidgetwindow.h"
#include "contourwidgetwindow.h"

#include <VtkUtils/vtkutils.h>
#include <VtkUtils/vtkwidget.h>

#include <vtkRenderWindow.h>
#include <vtkActor.h>
#include <vtkConeSource.h>
#include <vtkBoxWidget.h>
#include <vtkImplicitPlaneWidget.h>
#include <vtkSphereWidget.h>
#include <vtkSphereWidget2.h>
#include <vtkPointWidget.h>
#include <vtkLineWidget.h>
#include <vtkImagePlaneWidget.h>
#include <vtkPlaneWidget.h>
#include <vtkSplineWidget.h>
#include <vtkSliderWidget.h>
#include <vtkDistanceWidget.h>
#include <vtkAngleWidget.h>
#include <vtkCommand.h>
#include <vtkTransform.h>
#include <vtkCylinderSource.h>
#include <vtkProperty.h>

// ECV_DB_LIB
#include <ecvHObject.h>

using namespace DBLib;

QString SmallWidgetsInterface::getInterfaceName() const
{
	return QObject::tr("SmallWidgets Interface");
}

QStringList SmallWidgetsInterface::getWidgtsNames() const
{
	QStringList widgetsName;
	widgetsName << "ContourWidgetWindow" << "DistanceWidgetWindow" << "AngleWidgetWindow";
	return widgetsName;
}

QList<QWidget*> SmallWidgetsInterface::getWidgts() const
{
	if (m_widgetsMap.isEmpty())
	{
		return QList<QWidget*>();
	}
	return m_widgetsMap.values();
}


QWidget * SmallWidgetsInterface::getWidgtByID(const VTK_WIDGETS_TYPE widgetID) const
{
	if (!m_widgetsMap.contains(widgetID))
	{
		return nullptr;
	}
	return m_widgetsMap[widgetID];
}

const QString SmallWidgetsInterface::getWidgtNameByID(const VTK_WIDGETS_TYPE widgetID) const
{
	if (!m_NamesMap.contains(widgetID))
	{
		return QString();
	}
	return m_NamesMap[widgetID];
}

bool SmallWidgetsInterface::setInput(const ccHObject * obj, const VTK_WIDGETS_TYPE widgetType)
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

	BaseWidgetWindow * baseWidget = static_cast<BaseWidgetWindow *>(widget);
	if (!baseWidget)
	{
		return false;
	}

	return baseWidget->setInput(obj);
}

ccHObject * SmallWidgetsInterface::getOutput(const DBLib::VTK_WIDGETS_TYPE widgetType)
{
	if (m_widgetsMap.isEmpty() || m_NamesMap.isEmpty())
	{
		return nullptr;
	}

	BaseWidgetWindow * baseWidget = static_cast<BaseWidgetWindow *>(getWidgtByID(widgetType));
	if (!baseWidget)
	{
		return nullptr;
	}

	return baseWidget->getOutput();
}

void SmallWidgetsInterface::initWidgets()
{
	if (!m_widgetsMap.isEmpty() || !m_NamesMap.isEmpty())
	{
		return;
	}

	QStringList widgetNames = getWidgtsNames();

	for (QString & name : widgetNames)
	{
		if (name == QString("ContourWidgetWindow"))
		{
			m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_CONTOUR_WIDGET, new ContourWidgetWindow());
			m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_CONTOUR_WIDGET, name);
		}
		else if (name == QString("DistanceWidgetWindow"))
		{
			m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_DISTANCE_WIDGET, new DistanceWidgetWindow());
			m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_DISTANCE_WIDGET, name);
		}
		else if (name == QString("AngleWidgetWindow"))
		{
			m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_ANGLE_WIDGET, new AngleWidgetWindow());
			m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_ANGLE_WIDGET, name);
		}

	}
}

void SmallWidgetsInterface::unregister()
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
