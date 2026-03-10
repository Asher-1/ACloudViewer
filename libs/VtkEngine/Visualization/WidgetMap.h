// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file WidgetMap.h
/// @brief Maps view IDs to VTK actors and widgets for the PCL visualizer.

#include <vtkAbstractWidget.h>
#include <vtkLODActor.h>
#include <vtkSmartPointer.h>

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

template <typename T>
class vtkSmartPointer;
class vtkLODActor;
class vtkProp;

namespace Visualization {
/// @class WidgetMap
/// @brief Associates a VTK actor and optional widget with a view ID.
class WidgetMap {
public:
    WidgetMap() = default;

    virtual ~WidgetMap() {}

    /// VTK actor holding the data to render
    vtkSmartPointer<vtkLODActor> actor;
    /// Optional VTK widget (e.g., for interactive elements)
    vtkSmartPointer<vtkAbstractWidget> widget;
};

/// Map from view ID to WidgetMap
typedef std::unordered_map<std::string, WidgetMap> WidgetActorMap;
typedef std::shared_ptr<WidgetActorMap> WidgetActorMapPtr;

/// Map from view ID to VTK prop
typedef std::unordered_map<std::string, vtkSmartPointer<vtkProp>> PropActorMap;
typedef std::shared_ptr<PropActorMap> PropActorMapPtr;
}  // namespace Visualization
