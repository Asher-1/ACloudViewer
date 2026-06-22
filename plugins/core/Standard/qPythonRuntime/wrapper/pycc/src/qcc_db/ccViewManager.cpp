// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ecvGenericGLDisplay.h>
#include <ecvHObject.h>
#include <ecvViewManager.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccViewManager(py::module &m)
{
    py::class_<ecvViewManager> PyViewManager(m,
                                             "ccViewManager",
                                             R"doc(
Multi-view coordinator — tracks active view, source, and all registered views.

Mirrors ParaView's pqActiveObjects:
  - Active view tracking (getActiveView / setActiveView)
  - View registration (getAllViews / viewCount)
  - Batch operations (refreshAll / redrawAll)
  - Layout proxy management

Example:
    >>> vm = cloudViewer.ccViewManager.instance()
    >>> print(vm.viewCount())
    >>> view = vm.getActiveView()
    >>> vm.redrawAll()
)doc");

    PyViewManager
        .def_static("instance",
                    &ecvViewManager::instance,
                    py::return_value_policy::reference,
                    R"doc(Get the singleton instance.)doc")

        // Active view
        .def("getActiveView",
             &ecvViewManager::getActiveView,
             py::return_value_policy::reference,
             R"doc(Get the currently active view (may be None).)doc")
        .def("setActiveView",
             &ecvViewManager::setActiveView,
             "view"_a,
             R"doc(Set the active view.)doc")
        .def("getEffectiveView",
             &ecvViewManager::getEffectiveView,
             py::return_value_policy::reference,
             R"doc(Get the effective view (rendering override if set,
otherwise the UI-active view).)doc")

        // Active source
        .def("activeSource",
             &ecvViewManager::activeSource,
             py::return_value_policy::reference,
             R"doc(Get the active source entity.)doc")
        .def("setActiveSource",
             &ecvViewManager::setActiveSource,
             "source"_a,
             R"doc(Set the active source entity.)doc")

        // View query
        .def("getAllViews",
             &ecvViewManager::getAllViews,
             py::return_value_policy::reference,
             R"doc(Get the list of all registered views.)doc")
        .def(
            "viewCount", &ecvViewManager::viewCount, R"doc(Get the number of registered views.)doc")
        .def("findView",
             &ecvViewManager::findView,
             "uniqueID"_a,
             py::return_value_policy::reference,
             R"doc(Find a view by its unique ID.)doc")
        .def("getPrimaryView",
             &ecvViewManager::getPrimaryView,
             py::return_value_policy::reference,
             R"doc(Get the first registered (primary) view.)doc")
        .def("hasAnyView",
             &ecvViewManager::hasAnyView,
             R"doc(Returns True if at least one view is registered.)doc")

        // Batch operations
        .def("refreshAll",
             &ecvViewManager::refreshAll,
             "only2D"_a = false,
             R"doc(Refresh all registered views.)doc")
        .def("redrawAll",
             &ecvViewManager::redrawAll,
             "only2D"_a = false,
             "forceRedraw"_a = true,
             "includePrimary"_a = true,
             R"doc(Redraw all registered views.)doc");
}
