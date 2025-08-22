// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/Scalar.h"

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace core {

void pybind_core_scalar(py::module& m) {
    py::class_<Scalar> scalar(
            m, "Scalar", "A Scalar can store one of {double, int64, bool}.");

    scalar.def(py::init([](float val) { return Scalar(val); }));
    scalar.def(py::init([](double val) { return Scalar(val); }));
    scalar.def(py::init([](int8_t val) { return Scalar(val); }));
    scalar.def(py::init([](int16_t val) { return Scalar(val); }));
    scalar.def(py::init([](int32_t val) { return Scalar(val); }));
    scalar.def(py::init([](int64_t val) { return Scalar(val); }));
    scalar.def(py::init([](uint8_t val) { return Scalar(val); }));
    scalar.def(py::init([](uint16_t val) { return Scalar(val); }));
    scalar.def(py::init([](uint32_t val) { return Scalar(val); }));
    scalar.def(py::init([](uint64_t val) { return Scalar(val); }));
    scalar.def(py::init([](bool val) { return Scalar(val); }));
}

}  // namespace core
}  // namespace cloudViewer
