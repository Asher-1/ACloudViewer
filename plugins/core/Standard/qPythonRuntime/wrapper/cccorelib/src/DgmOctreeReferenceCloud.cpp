// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <DgmOctreeReferenceCloud.h>
#include <GenericIndexedCloudPersist.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_DgmOctreeReferenceCloud(py::module &cccorelib)
{
    py::class_<cloudViewer::DgmOctreeReferenceCloud, cloudViewer::GenericIndexedCloudPersist>(
        cccorelib, "DgmOctreeReferenceCloud")
        .def("forwardIterator", &cloudViewer::DgmOctreeReferenceCloud::forwardIterator);
}
