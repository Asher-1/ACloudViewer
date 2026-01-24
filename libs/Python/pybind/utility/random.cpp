// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/utility/Random.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/docstring.h"
#include "pybind/utility/utility.h"

namespace cloudViewer {
namespace utility {
namespace random {
void pybind_random(py::module &m) {
    py::module m_random =
            m.def_submodule("random", "Random number generation utilities.");

    m_random.def("seed", &Seed, "seed"_a,
                 "Set the global random seed for CloudViewer random number "
                 "generation.");
    docstring::FunctionDocInject(m_random, "seed",
                                 {{"seed", "The seed value to set."}});
}

}  // namespace random
}  // namespace utility
}  // namespace cloudViewer
