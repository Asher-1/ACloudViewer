// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "visualization/visualizer/Visualizer.h"
#include "visualization/visualizer/ViewControl.h"

#include "pybind/cloudViewer_pybind.h"

using namespace cloudViewer;

template <class VisualizerBase = visualization::Visualizer>
class PyVisualizer : public VisualizerBase {
public:
    using VisualizerBase::VisualizerBase;
    bool AddGeometry(std::shared_ptr<const ccHObject> geometry_ptr,
                     bool reset_bounding_box = true) override {
        PYBIND11_OVERLOAD(bool, VisualizerBase, AddGeometry, geometry_ptr);
    }
    bool UpdateGeometry(std::shared_ptr<const ccHObject> geometry_ptr =
                                nullptr) override {
        PYBIND11_OVERLOAD(bool, VisualizerBase, UpdateGeometry, );
    }
    bool HasGeometry() const override {
        PYBIND11_OVERLOAD(bool, VisualizerBase, HasGeometry, );
    }
    void UpdateRender() override {
        PYBIND11_OVERLOAD(void, VisualizerBase, UpdateRender, );
    }
    void PrintVisualizerHelp() override {
        PYBIND11_OVERLOAD(void, VisualizerBase, PrintVisualizerHelp, );
    }
    void UpdateWindowTitle() override {
        PYBIND11_OVERLOAD(void, VisualizerBase, UpdateWindowTitle, );
    }
    void BuildUtilities() override {
        PYBIND11_OVERLOAD(void, VisualizerBase, BuildUtilities, );
    }
};

template <class ViewControlBase = visualization::ViewControl>
class PyViewControl : public ViewControlBase {
public:
    using ViewControlBase::ViewControlBase;
    void Reset() override { PYBIND11_OVERLOAD(void, ViewControlBase, Reset, ); }
    void ChangeFieldOfView(double step) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, ChangeFieldOfView, step);
    }
    void ChangeWindowSize(int width, int height) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, ChangeWindowSize, width,
                          height);
    }
    void Scale(double scale) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, Scale, scale);
    }
    void Rotate(double x, double y, double xo, double yo) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, Rotate, x, y, xo, yo);
    }
    void Translate(double x, double y, double xo, double yo) override {
        PYBIND11_OVERLOAD(void, ViewControlBase, Translate, x, y, xo, yo);
    }
};
