// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
