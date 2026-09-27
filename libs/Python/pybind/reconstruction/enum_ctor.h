// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cloudViewer {
namespace reconstruction {

/// Upstream pycolmap parity (src/pycolmap/helpers.h
/// AddStringToEnumConstructor): adds a name-based constructor to a bound
/// enum so that python code can write ``Enum("MEMBER")`` in addition to
/// ``Enum.MEMBER``. str() round-trips through this constructor, while repr()
/// keeps the qualified "Type.MEMBER" form.
template <typename T>
inline void AddStringToEnumConstructor(py::enum_<T>& enm) {
    enm.def(
            py::init([enm](const std::string& name) {
                py::object member = enm.attr("__members__")[py::str(name)];
                return member.cast<T>();
            }),
            py::arg("name"),
            "Construct the enum member from its plain member name.");
}

}  // namespace reconstruction
}  // namespace cloudViewer
