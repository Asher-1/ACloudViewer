// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "widgetsFiltersInterface.h"

#include "clipwindow.h"
#include "decimatewindow.h"
#include "glyphwindow.h"
#include "isosurfacewindow.h"
#include "probewindow.h"
#include "slicewindow.h"
#include "smoothwindow.h"
#include "streamlinewindow.h"
#include "thresholdwindow.h"

// ECV_DB_LIB
#include <ecvHObject.h>

using namespace DBLib;

QString FiltersWidgetInterface::getInterfaceName() const {
    return QObject::tr("Filters Interface");
}

QStringList FiltersWidgetInterface::getWidgtsNames() const {
    QStringList widgetsName;
    widgetsName << "clipWidget"
                << "sliceWidget"
                << "isosurfaceWidget"
                << "thresholdWidget"
                << "streamlineWidget"
                << "glyphWidget"
                << "probeWidget"
                << "smootWidget"
                << "decimateWidget";
    return widgetsName;
}

QList<QWidget *> FiltersWidgetInterface::getWidgts() const {
    if (m_widgetsMap.isEmpty()) {
        return QList<QWidget *>();
    }
    return m_widgetsMap.values();
}

QWidget *FiltersWidgetInterface::getWidgtByID(
        const VTK_WIDGETS_TYPE widgetID) const {
    if (!m_widgetsMap.contains(widgetID)) {
        return nullptr;
    }
    return m_widgetsMap[widgetID];
}

const QString FiltersWidgetInterface::getWidgtNameByID(
        const VTK_WIDGETS_TYPE widgetID) const {
    if (!m_NamesMap.contains(widgetID)) {
        return QString();
    }
    return m_NamesMap[widgetID];
}

bool FiltersWidgetInterface::setInput(const ccHObject *obj,
                                      const VTK_WIDGETS_TYPE widgetType) {
    if (!obj || m_widgetsMap.isEmpty() || m_NamesMap.isEmpty()) {
        return false;
    }

    QWidget *widget = getWidgtByID(widgetType);
    if (!widget) {
        return false;
    }

    FilterWindow *filterWin = static_cast<FilterWindow *>(widget);
    if (!filterWin) {
        return false;
    }

    return filterWin->setInput(obj);
}

ccHObject *FiltersWidgetInterface::getOutput(
        const DBLib::VTK_WIDGETS_TYPE widgetType) {
    if (m_widgetsMap.isEmpty() || m_NamesMap.isEmpty()) {
        return nullptr;
    }

    FilterWindow *filterWin =
            static_cast<FilterWindow *>(getWidgtByID(widgetType));
    if (!filterWin) {
        return nullptr;
    }

    if (filterWin->windowTitle() == QString("Slice")) {
        SliceWindow *swin = static_cast<SliceWindow *>(filterWin);
        assert(swin);
        swin->setOutputMode(true);
        swin->apply();
    }

    return filterWin->getOutput();
}

void FiltersWidgetInterface::initWidgets() {
    if (!m_widgetsMap.isEmpty() || !m_NamesMap.isEmpty()) {
        return;
    }

    QStringList widgetNames = getWidgtsNames();

    for (QString &name : widgetNames) {
        if (name == QString("clipWidget")) {
            m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_CLIP_WIDGET,
                                new ClipWindow);
            m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_CLIP_WIDGET, name);
        } else if (name == QString("sliceWidget")) {
            m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_SLICE_WIDGET,
                                new SliceWindow);
            m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_SLICE_WIDGET, name);
        }
        /*else if (name == QString("isosurfaceWidget"))
        {
                m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_ISOSURFACE_WIDGET, new
        IsosurfaceWindow);
                m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_ISOSURFACE_WIDGET,
        name);
        }
        else if (name == QString("thresholdWidget"))
        {
                m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_THRESHOLD_WIDGET, new
        ThresholdWindow);
                m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_THRESHOLD_WIDGET, name);
        }
        else if (name == QString("streamlineWidget"))
        {
                m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_STREAMLINE_WIDGET, new
        StreamlineWindow);
                m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_STREAMLINE_WIDGET,
        name);
        }
        else if (name == QString("glyphWidget"))
        {
                m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_GLYPH_WIDGET, new
        GlyphWindow); m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_GLYPH_WIDGET,
        name);
        }
        else if (name == QString("probeWidget"))
        {
                m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_PROB_WIDGET, new
        ProbeWindow); m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_PROB_WIDGET,
        name);

        }
        else if (name == QString("smootWidget"))
        {
                m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_SMOOTH_WIDGET, new
        SmoothWindow); m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_SMOOTH_WIDGET,
        name);
        }
        else if (name == QString("decimateWidget"))
        {
                m_widgetsMap.insert(VTK_WIDGETS_TYPE::VTK_DECIMATEWIDGET_WIDGET,
        new DecimateWindow);
                m_NamesMap.insert(VTK_WIDGETS_TYPE::VTK_DECIMATEWIDGET_WIDGET,
        name);
        }*/
    }
}

void FiltersWidgetInterface::unregister() {
    if (m_widgetsMap.isEmpty()) {
        return;
    }

    QMap<VTK_WIDGETS_TYPE, QWidget *>::iterator iter = m_widgetsMap.begin();
    while (iter != m_widgetsMap.end()) {
        QWidget *widget = iter.value();
        if (widget) {
            // widget->deleteLater();
            delete widget;
            widget = nullptr;
        }
        iter++;
    }

    m_widgetsMap.clear();
    m_NamesMap.clear();
}
