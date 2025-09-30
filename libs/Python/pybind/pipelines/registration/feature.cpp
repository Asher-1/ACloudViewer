// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <ecvFeature.h>
#include <ecvPointCloud.h>

#include "pybind/docstring.h"
#include "pybind/utility/utility.h"

namespace cloudViewer {
namespace pipelines {
namespace registration {

void pybind_feature(py::module &m) {
    // cloudViewer.utility.Feature
    py::class_<utility::Feature, std::shared_ptr<utility::Feature>> feature(
            m, "Feature", "Class to store featrues for registration.");
    py::detail::bind_default_constructor<utility::Feature>(feature);
    py::detail::bind_copy_functions<utility::Feature>(feature);
    feature.def("resize", &utility::Feature::Resize, "dim"_a, "n"_a,
                "Resize feature data buffer to ``dim x n``.")
            .def("dimension", &utility::Feature::Dimension,
                 "Returns feature dimensions per point.")
            .def("num", &utility::Feature::Num, "Returns number of points.")
            .def("select_by_index", &utility::Feature::SelectByIndex,
                 "Function to select features from input Feature group into "
                 "output Feature group.",
                 "indices"_a, "invert"_a = false)
            .def_readwrite("data", &utility::Feature::data_,
                           "``dim x n`` float64 numpy array: Data buffer "
                           "storing features.")
            .def("__repr__", [](const utility::Feature &f) {
                return std::string(
                               "Feature class with "
                               "dimension "
                               "= ") +
                       std::to_string(f.Dimension()) +
                       std::string(" and num = ") + std::to_string(f.Num()) +
                       std::string("\nAccess its data via data member.");
            });
    docstring::ClassMethodDocInject(m, "Feature", "dimension");
    docstring::ClassMethodDocInject(m, "Feature", "num");
    docstring::ClassMethodDocInject(m, "Feature", "resize",
                                    {{"dim", "Feature dimension per point."},
                                     {"n", "Number of points."}});
    docstring::ClassMethodDocInject(
            m, "Feature", "select_by_index",
            {{"indices", "Indices of features to be selected."},
             {"invert",
              "Set to ``True`` to invert the selection of indices."}});
}

void pybind_feature_methods(py::module &m) {
    m.def("compute_fpfh_feature", &utility::ComputeFPFHFeature,
          "Function to compute FPFH feature for a point cloud", "input"_a,
          "search_param"_a, "indices"_a = py::none());
    docstring::FunctionDocInject(
            m, "compute_fpfh_feature",
            {
                    {"input", "The Input point cloud."},
                    {"search_param", "KDTree KNN search parameter."},
                    {"indices",
                     "Indices to select points for feature computation."},
            });

    m.def("correspondences_from_features",
          &utility::CorrespondencesFromFeatures,
          "Function to find nearest neighbor correspondences from features",
          "source_features"_a, "target_features"_a, "mutual_filter"_a = false,
          "mutual_consistency_ratio"_a = 0.1f);
    docstring::FunctionDocInject(
            m, "correspondences_from_features",
            {{"source_features", "The source features stored in (dim, N)."},
             {"target_features", "The target features stored in (dim, M)."},
             {"mutual_filter",
              "filter correspondences and return the collection of (i, j) s.t. "
              "source_features[i] and target_features[j] are mutually the "
              "nearest neighbor."},
             {"mutual_consistency_ratio",
              "Threshold to decide whether the number of filtered "
              "correspondences is sufficient. Only used when mutual_filter is "
              "enabled."}});
}

}  // namespace registration
}  // namespace pipelines
}  // namespace cloudViewer
