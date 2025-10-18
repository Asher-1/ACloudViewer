// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ml/contrib/contrib_nns.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/docstring.h"
#include "pybind/ml/contrib/contrib.h"
#include "pybind/pybind_utils.h"

namespace cloudViewer {
namespace ml {
namespace contrib {

void pybind_contrib_nns(py::module& m_contrib) {
    m_contrib.def("knn_search", &KnnSearch, "query_points"_a,
                  "dataset_points"_a, "knn"_a);
    m_contrib.def("radius_search", &RadiusSearch, "query_points"_a,
                  "dataset_points"_a, "query_batches"_a, "dataset_batches"_a,
                  "radius"_a);
}

}  // namespace contrib
}  // namespace ml
}  // namespace cloudViewer
