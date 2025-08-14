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
	py::class_<utility::Feature, std::shared_ptr<utility::Feature>>
		feature(m, "Feature", "Class to store featrues for registration.");
	py::detail::bind_default_constructor<utility::Feature>(feature);
	py::detail::bind_copy_functions<utility::Feature>(feature);
	feature.def("resize", &utility::Feature::Resize, "dim"_a, "n"_a,
		"Resize feature data buffer to ``dim x n``.")
		.def("dimension", &utility::Feature::Dimension,
			"Returns feature dimensions per point.")
		.def("num", &utility::Feature::Num,
			"Returns number of points.")
		.def_readwrite("data", &utility::Feature::data_,
			"``dim x n`` float64 numpy array: Data buffer "
			"storing features.")
		.def("__repr__", [](const utility::Feature &f) {
		return std::string(
			"utility::Feature class with dimension "
			"= ") +
			std::to_string(f.Dimension()) +
			std::string(" and num = ") + std::to_string(f.Num()) +
			std::string("\nAccess its data via data member.");
	});
	docstring::ClassMethodDocInject(m, "Feature", "dimension");
	docstring::ClassMethodDocInject(m, "Feature", "num");
	docstring::ClassMethodDocInject(m, "Feature", "resize",
		{ {"dim", "Feature dimension per point."},
		 {"n", "Number of points."} });
}

void pybind_feature_methods(py::module &m) {
	m.def("compute_fpfh_feature", &utility::ComputeFPFHFeature,
		"Function to compute FPFH feature for a point cloud", "input"_a,
		"search_param"_a);
	docstring::FunctionDocInject(
		m, "compute_fpfh_feature",
		{ {"input", "The Input point cloud."},
		 {"search_param", "KDTree KNN search parameter."} });
}

}  // namespace registration
}  // namespace pipelines
}  // namespace cloudViewer
