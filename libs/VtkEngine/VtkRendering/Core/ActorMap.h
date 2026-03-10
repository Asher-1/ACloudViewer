// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file ActorMap.h
 *  @brief Actor and widget map types for VTK cloud/shape/coordinate rendering
 */

#include <vtkIdTypeArray.h>
#include <vtkLODActor.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

#include <memory>
#include <string>
#include <unordered_map>

class vtkAbstractWidget;
class vtkProp;

namespace VtkRendering {

/** @struct CloudActor
 *  @brief Represents a cloud actor with its VTK rendering state.
 *  Replaces pcl::visualization::CloudActor with a PCL-free implementation.
 */
struct CloudActor {
    CloudActor() = default;
    ~CloudActor() = default;

    vtkSmartPointer<vtkLODActor> actor;
    vtkSmartPointer<vtkMatrix4x4> viewpoint_transformation;
    vtkSmartPointer<vtkIdTypeArray> cells;
};

/** @struct WidgetActor
 *  @brief Represents a widget actor with optional VTK widget.
 */
struct WidgetActor {
    WidgetActor() = default;
    ~WidgetActor() = default;

    vtkSmartPointer<vtkLODActor> actor;
    vtkSmartPointer<vtkAbstractWidget> widget;
};

using CloudActorMap = std::unordered_map<std::string, CloudActor>;
using CloudActorMapPtr = std::shared_ptr<CloudActorMap>;

using ShapeActorMap = std::unordered_map<std::string, vtkSmartPointer<vtkProp>>;
using ShapeActorMapPtr = std::shared_ptr<ShapeActorMap>;

using CoordinateActorMap =
        std::unordered_map<std::string, vtkSmartPointer<vtkProp>>;
using CoordinateActorMapPtr = std::shared_ptr<CoordinateActorMap>;

using WidgetActorMap = std::unordered_map<std::string, WidgetActor>;
using WidgetActorMapPtr = std::shared_ptr<WidgetActorMap>;

using PropActorMap = std::unordered_map<std::string, vtkSmartPointer<vtkProp>>;
using PropActorMapPtr = std::shared_ptr<PropActorMap>;

}  // namespace VtkRendering
