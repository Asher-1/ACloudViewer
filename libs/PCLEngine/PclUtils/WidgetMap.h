// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

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

namespace PclUtils {
class WidgetMap {
public:
    WidgetMap() = default;

    virtual ~WidgetMap() {}

    /** \brief The actor holding the data to render. */
    vtkSmartPointer<vtkLODActor> actor;
    vtkSmartPointer<vtkAbstractWidget> widget;
};

typedef std::unordered_map<std::string, WidgetMap> WidgetActorMap;
typedef std::shared_ptr<WidgetActorMap> WidgetActorMapPtr;

typedef std::unordered_map<std::string, vtkSmartPointer<vtkProp>> PropActorMap;
typedef std::shared_ptr<PropActorMap> PropActorMapPtr;
}  // namespace PclUtils
