// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvWidgetsInterface.h"

#include <CVLog.h>

// system
#include <cassert>
#include <vector>

// QT
#include <QWidget>

namespace DBLib {

//! Available filters
/** Filters are uniquely recognized by their 'file filter' string.
        We use a std::vector so as to keep the insertion ordering!
**/
static ecvWidgetsInterface::InterfaceContainer s_widgetInterfaces;

QWidget* ecvWidgetsInterface::LoadWidget(const VTK_WIDGETS_TYPE widgetType,
                                         Shared widgetInterface) {
    if (!widgetInterface) {
        CVLog::Error(QString("[Load] Internal error (invalid input widget type "
                             ": %1)")
                             .arg(widgetType));
        assert(false);
        return nullptr;
    }

    QWidget* widget = widgetInterface->getWidgtByID(widgetType);
    if (!widget) {
        return nullptr;
    }
    return widget;
}

QWidget* ecvWidgetsInterface::LoadWidget(const VTK_WIDGETS_TYPE widgetType) {
    Shared widgetInterface(nullptr);
    widgetInterface = GetWigetInterface(widgetType);
    if (!widgetInterface) {
        CVLog::Error(QString("[Load] Internal error: no widget interface "
                             "corresponds to widget type '%1'")
                             .arg(widgetType));
        return nullptr;
    }

    return LoadWidget(widgetType, widgetInterface);
}

bool ecvWidgetsInterface::SetInput(const ccHObject* obj,
                                   const VTK_WIDGETS_TYPE widgetType) {
    Shared widgetInterface(nullptr);
    widgetInterface = GetWigetInterface(widgetType);
    if (!widgetInterface) {
        CVLog::Error(QString("[Load] Internal error: no widget interface "
                             "corresponds to widget type '%1'")
                             .arg(widgetType));
        return false;
    }
    return widgetInterface->setInput(obj, widgetType);
}

ccHObject* ecvWidgetsInterface::GetOutput(const VTK_WIDGETS_TYPE widgetType) {
    Shared widgetInterface(nullptr);
    widgetInterface = GetWigetInterface(widgetType);
    if (!widgetInterface) {
        CVLog::Error(QString("[Load] Internal error: no widget interface "
                             "corresponds to widget type '%1'")
                             .arg(widgetType));
        return nullptr;
    }
    return widgetInterface->getOutput(widgetType);
}

void ecvWidgetsInterface::InitInternalInterfaces() {
    // from the most useful to the less one!
    // Register(Shared(new BinFilter()));
}

void ecvWidgetsInterface::Register(Shared widgetInterface) {
    if (!widgetInterface) {
        assert(false);
        return;
    }

    // filters are uniquely recognized by their 'file filter' string
    const QString interfaceName = widgetInterface->getInterfaceName();
    const QStringList widgetName = widgetInterface->getWidgtsNames();
    for (InterfaceContainer::const_iterator it = s_widgetInterfaces.begin();
         it != s_widgetInterfaces.end(); ++it) {
        bool error = false;
        if (*it == widgetInterface) {
            CVLog::Warning(
                    QStringLiteral("[ecvWidgetsInterface::Register] widget "
                                   "Interface '%1' is already registered")
                            .arg(interfaceName));
            error = true;
        } else {
            // we are going to compare the widgets as they should remain unique!
            const QStringList otherInterfaceWidgets = (*it)->getWidgtsNames();
            for (int i = 0; i < widgetName.size(); ++i) {
                if (otherInterfaceWidgets.contains(widgetName[i])) {
                    const QString otherInterfaceName =
                            widgetInterface->getInterfaceName();
                    CVLog::Warning(
                            QStringLiteral("[ecvWidgetsInterface::Register] "
                                           "Internal error: widget '%1' of  "
                                           "widget Interface '%2' "
                                           "is already handled by another "
                                           "widget interface ('%3')!")
                                    .arg(widgetName[i], interfaceName,
                                         otherInterfaceName));
                    error = true;
                    break;
                }
            }
        }

        if (error) return;
    }

    // insert filter
    widgetInterface->initWidgets();
    s_widgetInterfaces.push_back(widgetInterface);
}

void ecvWidgetsInterface::UnregisterAll() {
    for (InterfaceContainer::iterator it = s_widgetInterfaces.begin();
         it != s_widgetInterfaces.end(); ++it) {
        (*it)->unregister();
    }
    s_widgetInterfaces.clear();
}

ecvWidgetsInterface::Shared ecvWidgetsInterface::GetWigetInterface(
        const VTK_WIDGETS_TYPE widgetID) {
    for (InterfaceContainer::const_iterator it = s_widgetInterfaces.begin();
         it != s_widgetInterfaces.end(); ++it) {
        if ((*it)->getWidgtByID(widgetID)) return *it;
    }

    return Shared(nullptr);
}

const ecvWidgetsInterface::InterfaceContainer&
ecvWidgetsInterface::GetWigetInterfaces() {
    return s_widgetInterfaces;
}
}  // namespace DBLib
