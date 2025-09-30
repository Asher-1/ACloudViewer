// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace core {
class Tensor;
}
}  // namespace cloudViewer

// Define type caster allowing implicit conversion to Tensor from common types.
// Needs to be included in each compilation unit.
namespace pybind11 {
namespace detail {
template <>
struct type_caster<cloudViewer::core::Tensor>
    : public type_caster_base<cloudViewer::core::Tensor> {
public:
    bool load(handle src, bool convert);

private:
    std::unique_ptr<cloudViewer::core::Tensor> holder_;
};

}  // namespace detail
}  // namespace pybind11
