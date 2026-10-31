// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// ScalarContainer must not use a copying list caster in another binding translation unit.
#include "../wrapper/pycc/src/casters.h"
#include <GenericDistribution.h>
#include <pybind11/stl.h>

bool consumer_accepts_scalar_list(pybind11::handle value)
{
    pybind11::detail::type_caster<cloudViewer::GenericDistribution::ScalarContainer> caster;
    return caster.load(value, true);
}

void consumer_mutates_scalar_container(pybind11::handle value)
{
    auto &values = value.cast<cloudViewer::GenericDistribution::ScalarContainer &>();
    values.at(0) = ScalarType(7.0);
}
