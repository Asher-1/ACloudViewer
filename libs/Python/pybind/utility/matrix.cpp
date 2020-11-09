#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "Utility/Matrix.h"

#include "pybind/docstring.h"
#include "pybind/cloudViewer_pybind.h"

namespace py = pybind11;

namespace cloudViewer {
namespace utility {

void pybind_matrix(py::module &m) {
	// cloudViewer.registration.TransformationEstimationPointToPoint:
	// TransformationEstimation
	py::class_<utility::Matrix<PointCoordinateType>>
		te_p2p(m, "Matrix", "Class to interface of numpy and std::vector.");

	py::detail::bind_default_constructor< utility::Matrix<PointCoordinateType> >(te_p2p);
	py::detail::bind_copy_functions< utility::Matrix<PointCoordinateType> >(te_p2p);

	te_p2p.def(py::init([](const std::vector<size_t> &shape, const PointCoordinateType *data) {
		return new utility::Matrix<PointCoordinateType>(shape, data);
	}),
		"shape"_a = std::vector<size_t>(0), "data"_a = NULL)
		.def("__repr__",
			[](const utility::Matrix<PointCoordinateType> &te) {
		return std::string("utility::Matrix");
	})
		.def("data",
			[](const utility::Matrix<PointCoordinateType> &s) {
		return s.data();
	},
			"Function to get matrix internal ptr")
		.def("shape",
			[](const utility::Matrix<PointCoordinateType> &s, size_t ndim = 0) {
		return s.shape(ndim);
	},
			"ndim"_a, "Function to get matrix shape")
		.def("strides",
			[](const utility::Matrix<PointCoordinateType> &s, bool bytes = false) {
		return s.strides(bytes);
	},
			"bytes"_a, "Function to get matrix strides")
		.def("ndim",
			[](const utility::Matrix<PointCoordinateType> &s) {
		return s.ndim();
	},
			"Function to get matrix dimension")
		.def("size",
			[](const utility::Matrix<PointCoordinateType> &s) {
		return s.size();
	}, "Function to get matrix size");
}

}  // namespace utility
}  // namespace cloudViewer
