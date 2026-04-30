// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvBBox.h>
#include <ecvDisplayTools.h>
#include <ecvGenericGLDisplay.h>
#include <ecvHObject.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccDisplayTools(py::module &m)
{
    py::class_<ecvDisplayTools, ecvGenericDisplayTools> PyccDisplayTools(m, "ccDisplayTools");
    py::enum_<ecvGenericGLDisplay::PICKING_MODE> PyPickingMode(PyccDisplayTools, "PICKING_MODE");
    py::enum_<ecvGenericGLDisplay::INTERACTION_FLAG> PyInteractionFlag(
        PyccDisplayTools, "INTERACTION_FLAG", py::arithmetic());
    py::enum_<ecvGenericGLDisplay::MessagePosition> PyMessagePosition(PyccDisplayTools,
                                                                      "MessagePosition");
    py::enum_<ecvGenericGLDisplay::MessageType> PyMessageType(PyccDisplayTools, "MessageType");
    py::enum_<ecvGenericGLDisplay::PivotVisibility> PyPivotVisibility(PyccDisplayTools,
                                                                      "PivotVisibility");

    PyPickingMode.value("NO_PICKING", ecvGenericGLDisplay::PICKING_MODE::NO_PICKING)
        .value("ENTITY_PICKING", ecvGenericGLDisplay::PICKING_MODE::ENTITY_PICKING)
        .value("ENTITY_RECT_PICKING", ecvGenericGLDisplay::PICKING_MODE::ENTITY_RECT_PICKING)
        .value("FAST_PICKING", ecvGenericGLDisplay::PICKING_MODE::FAST_PICKING)
        .value("POINT_PICKING", ecvGenericGLDisplay::PICKING_MODE::POINT_PICKING)
        .value("TRIANGLE_PICKING", ecvGenericGLDisplay::PICKING_MODE::TRIANGLE_PICKING)
        .value("POINT_OR_TRIANGLE_PICKING",
               ecvGenericGLDisplay::PICKING_MODE::POINT_OR_TRIANGLE_PICKING)
        .value("POINT_OR_TRIANGLE_OR_LABEL_PICKING",
               ecvGenericGLDisplay::PICKING_MODE::POINT_OR_TRIANGLE_OR_LABEL_PICKING)
        .value("LABEL_PICKING", ecvGenericGLDisplay::PICKING_MODE::LABEL_PICKING)
        .value("DEFAULT_PICKING", ecvGenericGLDisplay::PICKING_MODE::DEFAULT_PICKING)
        .export_values();

    PyInteractionFlag
        .value("INTERACT_NONE", ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_NONE)
        .value("INTERACT_ROTATE",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_ROTATE)
        .value("INTERACT_PAN", ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_PAN)
        .value("INTERACT_CTRL_PAN",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_CTRL_PAN)
        .value("INTERACT_ZOOM_CAMERA",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_ZOOM_CAMERA)
        .value("INTERACT_2D_ITEMS",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_2D_ITEMS)
        .value("INTERACT_CLICKABLE_ITEMS",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_CLICKABLE_ITEMS)
        .value("INTERACT_TRANSFORM_ENTITIES",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_TRANSFORM_ENTITIES)
        .value("INTERACT_SIG_RB_CLICKED",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_SIG_RB_CLICKED)
        .value("INTERACT_SIG_LB_CLICKED",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_SIG_LB_CLICKED)
        .value("INTERACT_SIG_MOUSE_MOVED",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_SIG_MOUSE_MOVED)
        .value("INTERACT_SIG_BUTTON_RELEASED",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_SIG_BUTTON_RELEASED)
        .value("INTERACT_SIG_MB_CLICKED",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_SIG_MB_CLICKED)
        .value("INTERACT_SEND_ALL_SIGNALS",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::INTERACT_SEND_ALL_SIGNALS)
        .value("MODE_PAN_ONLY", ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::MODE_PAN_ONLY)
        .value("MODE_TRANSFORM_CAMERA",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::MODE_TRANSFORM_CAMERA)
        .value("MODE_TRANSFORM_ENTITIES",
               ecvGenericGLDisplay::INTERACTION_FLAGS::enum_type::MODE_TRANSFORM_ENTITIES)
        .export_values();

    PyMessagePosition
        .value("LOWER_LEFT_MESSAGE", ecvGenericGLDisplay::MessagePosition::LOWER_LEFT_MESSAGE)
        .value("UPPER_CENTER_MESSAGE", ecvGenericGLDisplay::MessagePosition::UPPER_CENTER_MESSAGE)
        .value("SCREEN_CENTER_MESSAGE", ecvGenericGLDisplay::MessagePosition::SCREEN_CENTER_MESSAGE)
        .export_values();

    PyMessageType.value("CUSTOM_MESSAGE", ecvGenericGLDisplay::MessageType::CUSTOM_MESSAGE)
        .value("SCREEN_SIZE_MESSAGE", ecvGenericGLDisplay::MessageType::SCREEN_SIZE_MESSAGE)
        .value("PERSPECTIVE_STATE_MESSAGE",
               ecvGenericGLDisplay::MessageType::PERSPECTIVE_STATE_MESSAGE)
        .value("SUN_LIGHT_STATE_MESSAGE", ecvGenericGLDisplay::MessageType::SUN_LIGHT_STATE_MESSAGE)
        .value("CUSTOM_LIGHT_STATE_MESSAGE",
               ecvGenericGLDisplay::MessageType::CUSTOM_LIGHT_STATE_MESSAGE)
        .value("MANUAL_TRANSFORMATION_MESSAGE",
               ecvGenericGLDisplay::MessageType::MANUAL_TRANSFORMATION_MESSAGE)
        .value("MANUAL_SEGMENTATION_MESSAGE",
               ecvGenericGLDisplay::MessageType::MANUAL_SEGMENTATION_MESSAGE)
        .value("ROTAION_LOCK_MESSAGE", ecvGenericGLDisplay::MessageType::ROTAION_LOCK_MESSAGE)
        .value("FULL_SCREEN_MESSAGE", ecvGenericGLDisplay::MessageType::FULL_SCREEN_MESSAGE)
        .export_values();

    PyPivotVisibility.value("PIVOT_HIDE", ecvGenericGLDisplay::PivotVisibility::PIVOT_HIDE)
        .value("PIVOT_SHOW_ON_MOVE", ecvGenericGLDisplay::PivotVisibility::PIVOT_SHOW_ON_MOVE)
        .value("PIVOT_ALWAYS_SHOW", ecvGenericGLDisplay::PivotVisibility::PIVOT_ALWAYS_SHOW)
        .export_values();

    PyccDisplayTools.def_static("getDevicePixelRatio", &ecvDisplayTools::GetDevicePixelRatio)
        .def_static(
            "doResize", static_cast<void (*)(int, int)>(&ecvDisplayTools::DoResize), "x"_a, "y"_a)
        .def_static(
            "doResize", static_cast<void (*)(const QSize &)>(&ecvDisplayTools::DoResize), "size"_a)
        .def_static("setSceneDB", &ecvDisplayTools::SetSceneDB, "root"_a)
        .def_static("getSceneDB", &ecvDisplayTools::GetSceneDB)
        .def_static(
            "renderText",
            [](int x,
               int y,
               const QString &str,
               const QFont &font,
               const ecvColor::Rgbub &color,
               const QString &id) { ecvDisplayTools::RenderText(x, y, str, font, color, id); },
            "x"_a,
            "y"_a,
            "str"_a,
            "font"_a = QFont(),
            "color"_a = ecvColor::defaultLabelBkgColor,
            "id"_a = "")
        .def_static(
            "renderText",
            [](double x,
               double y,
               double z,
               const QString &str,
               const QFont &font,
               const ecvColor::Rgbub &color,
               const QString &id) { ecvDisplayTools::RenderText(x, y, z, str, font, color, id); },
            "x"_a,
            "y"_a,
            "z"_a,
            "str"_a,
            "font"_a = QFont(),
            "color"_a = ecvColor::defaultLabelBkgColor,
            "id"_a = "")
        // TODO as widget
        .def_static("getScreenSize", &ecvDisplayTools::GetScreenSize)
        .def_static(
            "getGLCameraParameters",
            static_cast<void (*)(ccGLCameraParameters &)>(&ecvDisplayTools::GetGLCameraParameters),
            "params"_a)
        .def_static("displayNewMessage",
                    &ecvDisplayTools::DisplayNewMessage,
                    "message"_a,
                    "pos"_a,
                    "append"_a = false,
                    "displayMaxDelay_sec"_a = 2,
                    "type"_a = ecvGenericGLDisplay::MessageType::CUSTOM_MESSAGE)
        .def_static(
            "setPivotVisibility",
            [](ecvGenericGLDisplay::PivotVisibility vis)
            { ecvDisplayTools::SetPivotVisibility(vis); },
            "vis"_a)
        .def_static("getPivotVisibility",
                    static_cast<ecvGenericGLDisplay::PivotVisibility (*)()>(
                        &ecvDisplayTools::GetPivotVisibility))
        .def_static("showPivotSymbol", &ecvDisplayTools::ShowPivotSymbol, "state"_a)
        .def_static("setPivotPoint",
                    &ecvDisplayTools::SetPivotPoint,
                    "P"_a,
                    "autoUpdateCameraPos"_a = false,
                    "verbose"_a = false)
        .def_static("setCameraPos", &ecvDisplayTools::SetCameraPos, "P"_a)
        .def_static(
            "moveCamera", [](const CCVector3d &v) { ecvDisplayTools::MoveCamera(v); }, "v"_a)
        .def_static("setPerspectiveState",
                    &ecvDisplayTools::SetPerspectiveState,
                    "state"_a,
                    "objectCenteredView"_a)
        .def_static(
            "getPerspectiveState", &ecvDisplayTools::GetPerspectiveState, "objectCentered"_a)
        .def_static("objectPerspectiveEnabled", &ecvDisplayTools::ObjectPerspectiveEnabled)
        .def_static("viewerPerspectiveEnabled", &ecvDisplayTools::ViewerPerspectiveEnabled)
        .def_static("updateConstellationCenterAndZoom",
                    &ecvDisplayTools::UpdateConstellationCenterAndZoom,
                    "boundingBox"_a = nullptr,
                    "redraw"_a = true)
        .def_static("getVisibleObjectsBB",
                    &ecvDisplayTools::GetVisibleObjectsBB,
                    "box"_a,
                    py::arg("display") = nullptr)
        .def_static(
            "setView",
            [](CC_VIEW_ORIENTATION orientation, bool redraw)
            { ecvDisplayTools::SetView(orientation, redraw); },
            "orientation"_a,
            "redraw"_a = true)
        .def_static("setInteractionMode",
                    static_cast<void (*)(ecvGenericGLDisplay::INTERACTION_FLAGS)>(
                        &ecvDisplayTools::SetInteractionMode),
                    "flags"_a)
        .def_static("getInteractionMode",
                    static_cast<ecvGenericGLDisplay::INTERACTION_FLAGS (*)()>(
                        &ecvDisplayTools::GetInteractionMode))
        .def_static("setPickingMode",
                    static_cast<void (*)(ecvGenericGLDisplay::PICKING_MODE)>(
                        &ecvDisplayTools::SetPickingMode),
                    "mode"_a = ecvGenericGLDisplay::PICKING_MODE::DEFAULT_PICKING)
        .def_static(
            "getPickingMode",
            static_cast<ecvGenericGLDisplay::PICKING_MODE (*)()>(&ecvDisplayTools::GetPickingMode))
        .def_static("lockPickingMode", &ecvDisplayTools::LockPickingMode, "state"_a)
        .def_static("isPickingModeLocked", &ecvDisplayTools::IsPickingModeLocked)
        .def_static("getContext",
                    static_cast<void (*)(CC_DRAW_CONTEXT &)>(&ecvDisplayTools::GetContext),
                    "context"_a)
        // TODO static constexprs
        .def_static("setPointSize",
                    static_cast<void (*)(float, bool, int)>(&ecvDisplayTools::SetPointSize),
                    "size"_a,
                    "silent"_a = false,
                    "viewport"_a = 0)
        .def_static("setLineWidth",
                    static_cast<void (*)(float, bool, int)>(&ecvDisplayTools::SetLineWidth),
                    "width"_a,
                    "silent"_a = false,
                    "viewport"_a = 0)
        .def_static("getFontPointSize", &ecvDisplayTools::GetFontPointSize)
        .def_static("getLabelFontPointSize", &ecvDisplayTools::GetLabelFontPointSize)
        .def_static("getOwnDB", &ecvDisplayTools::GetOwnDB)
        .def_static("addToOwnDB", &ecvDisplayTools::AddToOwnDB, "obj"_a, "noDependency"_a = false)
        .def_static("removeFromOwnDB", &ecvDisplayTools::RemoveFromOwnDB, "obj"_a)
        .def_static(
            "setViewportParameters", &ecvDisplayTools::SetViewportParameters, "parameters"_a)
        .def_static("setFov", &ecvDisplayTools::SetFov, "fov"_a)
        .def_static("getFov", &ecvDisplayTools::GetFov)
        .def_static("invalidateVisualization", &ecvDisplayTools::InvalidateVisualization)
        // TODO renderToImage
        .def_static("renderToFile",
                    &ecvDisplayTools::RenderToFile,
                    "filename"_a,
                    "zoomFactor"_a = 1.0f,
                    "dontScaleFeatures"_a = false,
                    "renderOverlayItems"_a = false)
        .def_static("computeActualPixelSize", &ecvDisplayTools::ComputeActualPixelSize)
        .def_static("isRectangularPickingAllowed", &ecvDisplayTools::IsRectangularPickingAllowed)
        .def_static("setRectangularPickingAllowed",
                    &ecvDisplayTools::SetRectangularPickingAllowed,
                    "state"_a)
        // TODO: this would need a ccGui::ParamStruct binding
        //        .def_static("setShader", &ecvDisplayTools::getDisplayParameters)
        //        .def_static("setDisplayParameter", &ecvDisplayTools::getDisplayParameters,
        //        "params"_a, thisWindowOnly)
        .def_static("setPickingRadius", &ecvDisplayTools::SetPickingRadius, "radius"_a)
        .def_static("getPickingRadius", &ecvDisplayTools::GetPickingRadius)
        .def_static(
            "displayOverlayEntities", &ecvDisplayTools::DisplayOverlayEntities, "showScale"_a)
        // display
        .def_static(
            "getMainWindow", &ecvDisplayTools::GetMainWindow, py::return_value_policy::reference)
        .def_static("redrawDisplay",
                    &ecvDisplayTools::RedrawDisplay,
                    "only2D"_a = false,
                    "forceRedraw"_a = true)
        .def_static("refreshDisplay",
                    &ecvDisplayTools::RefreshDisplay,
                    "only2D"_a = false,
                    "forceRedraw"_a = true)

        .def_static("toBeRefreshed", &ecvDisplayTools::ToBeRefreshed)
        .def_static("invalidateViewport", &ecvDisplayTools::InvalidateViewport)
        .def_static("deprecate3DLayer", &ecvDisplayTools::Deprecate3DLayer)
        .def_static("getTextDisplayFont", &ecvDisplayTools::GetTextDisplayFont)
        .def_static("getLabelDisplayFont", &ecvDisplayTools::GetLabelDisplayFont)

        .def_static(
            "displayText",
            [](const QString &text,
               int x,
               int y,
               unsigned char align,
               float bkgAlpha,
               const unsigned char *color,
               const QFont *font,
               const QString &id)
            { ecvDisplayTools::DisplayText(text, x, y, align, bkgAlpha, color, font, id); },
            "text"_a,
            "x"_a,
            "y"_a,
            "align"_a = ecvGenericDisplayTools::ALIGN_DEFAULT,
            "bkgAlpha"_a = 0.0f,
            "color"_a = nullptr,
            "font"_a = nullptr,
            "id"_a = "")
        .def_static("display3DLabel",
                    &ecvDisplayTools::Display3DLabel,
                    "str"_a,
                    "pos3D"_a,
                    "color"_a = nullptr,
                    "font"_a = QFont())
        .def_static("remove3DLabel", &ecvDisplayTools::Remove3DLabel, "view_id"_a)
        .def_static("removeAllWidgets", &ecvDisplayTools::RemoveAllWidgets, "update"_a = true)
        .def_static(
            "toCenteredGLCoordinates", &ecvDisplayTools::ToCenteredGLCoordinates, "x"_a, "y"_a)
        .def_static("getViewportParameters", &ecvDisplayTools::GetViewportParameters)
        .def_static("setupProjectiveViewport",
                    &ecvDisplayTools::SetupProjectiveViewport,
                    "cameraMatrix"_a,
                    "fov_deg"_a = 0.0f,
                    "ar"_a = 1.0f,
                    "viewerBasedPerspective"_a = true,
                    "bubbleViewMode"_a = false)
        // TODO
        ;
}
