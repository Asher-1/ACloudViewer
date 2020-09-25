// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#include "scalarfield.h"
#include <ecvScalarField.h>
#include <ecvPointCloud.h>
#include "cloudViewer_pybind/docstring.h"
#include "cloudViewer_pybind/cloudViewer_pybind.h"

using namespace cloudViewer;
void pybind_scalarfield(py::module &m) {
	// cloudViewer.utility.ScalarField
	py::class_<CVLib::ScalarField, std::unique_ptr<CVLib::ScalarField>>
		scalarfieldBase(m, "ScalarField", 
			"A simple scalar field (to be associated to a point cloud).");
	//py::detail::bind_copy_functions<CVLib::ScalarField>(scalarfieldBase);
	scalarfieldBase.def(py::init([](const std::string& name) {
		return new CVLib::ScalarField(name.c_str());
	}), "name"_a = "ScalarField")
	.def("__repr__", [](const CVLib::ScalarField &sf) {
		std::string info = fmt::format(
			"ScalarField ({}) with {} scalars and range ({}, {}) ",
			sf.getName(), sf.currentSize(), sf.getMin(), sf.getMax());
		return info;
	})
	.def("set_name", [](CVLib::ScalarField &sf, 
						const std::string& name) {
		sf.setName(name.c_str());
	}, "Sets scalar field name.", "name"_a)
	.def("get_name", [](const CVLib::ScalarField &sf) {
			return std::string(sf.getName());
		}, "Returns scalar field name.")
	.def("compute_mean_variance", [](const CVLib::ScalarField &sf) {
			ScalarType mean, variance;
			sf.computeMeanAndVariance(mean, &variance);
			return std::make_tuple(mean, variance);
		}, "Computes the mean value (and optionally the variance value) of the scalar field.")
	.def("compute_min_max", &CVLib::ScalarField::computeMinAndMax, "Determines the min and max values")
	.def("invalid_value", &CVLib::ScalarField::flagValueAsInvalid,
		"Sets the value as 'invalid' (i.e. NAN_VALUE).", "index"_a)
	.def("get_min", &CVLib::ScalarField::getMin, "Returns the minimum value.")
	.def("get_max", &CVLib::ScalarField::getMax, "Returns the maximum value.")
	.def("fill", &CVLib::ScalarField::fill, "Returns the maximum value.", "fill_value"_a = 0)
	.def("reserve", &CVLib::ScalarField::reserveSafe, "Reserves memory (no exception thrown).", "count"_a)
	.def("resize", &CVLib::ScalarField::resizeSafe, "Resizes memory (no exception thrown).",
		"count"_a, "is_fill"_a = false, "value"_a = 0)
	.def("get_value", py::overload_cast<std::size_t>(&CVLib::ScalarField::getValue),
		"Gets value.", "index"_a)
	.def("get_value", py::overload_cast<std::size_t>(&CVLib::ScalarField::getValue, py::const_),
		"Gets value(const version).", "index"_a)
	.def("set_value", &CVLib::ScalarField::setValue, "Sets value.", "index"_a, "value"_a)
	.def("add_element", &CVLib::ScalarField::addElement, "Adds element.", "value"_a)
	.def("current_size", &CVLib::ScalarField::currentSize, "Returns current size.")
	.def("link", &CVLib::ScalarField::link, "Increase counter.")
	.def("release", &CVLib::ScalarField::release, "Decrease counter and deletes object when 0.")
	.def("get_link_count", &CVLib::ScalarField::getLinkCount, "Returns the current link count.")
	.def("swap", &CVLib::ScalarField::swap, "Swaps scalar value.", "i1"_a, "i2"_a)
	.def_static("Is_Valid_Value", &CVLib::ScalarField::ValidValue, "Returns whether a scalar value is valid or not.", "value"_a)
	.def_static("Nan", &CVLib::ScalarField::NaN, "Returns the specific NaN value");

	docstring::ClassMethodDocInject(m, "ScalarField", "link");
	docstring::ClassMethodDocInject(m, "ScalarField", "release");
	docstring::ClassMethodDocInject(m, "ScalarField", "get_link_count");
	docstring::ClassMethodDocInject(m, "ScalarField", "invalid_value");
	docstring::ClassMethodDocInject(m, "ScalarField", "set_name");
	docstring::ClassMethodDocInject(m, "ScalarField", "get_name");
	docstring::ClassMethodDocInject(m, "ScalarField", "get_min");
	docstring::ClassMethodDocInject(m, "ScalarField", "get_max");
	docstring::ClassMethodDocInject(m, "ScalarField", "compute_min_max");
	docstring::ClassMethodDocInject(m, "ScalarField", "compute_mean_variance");
	docstring::ClassMethodDocInject(m, "ScalarField", "fill");
	docstring::ClassMethodDocInject(m, "ScalarField", "reserve");
	docstring::ClassMethodDocInject(m, "ScalarField", "resize");
	docstring::ClassMethodDocInject(m, "ScalarField", "get_value");
	docstring::ClassMethodDocInject(m, "ScalarField", "set_value");
	docstring::ClassMethodDocInject(m, "ScalarField", "add_element");
	docstring::ClassMethodDocInject(m, "ScalarField", "current_size");
	docstring::ClassMethodDocInject(m, "ScalarField", "swap");

	// cloudViewer.utility.Range
	py::class_<ccScalarField::Range, std::shared_ptr<ccScalarField::Range>>
		range(m, "Range", "Scalar field range structure.");
	py::detail::bind_default_constructor<ccScalarField::Range>(range);
	py::detail::bind_copy_functions<ccScalarField::Range>(range);
	range.def(py::init<>())
	.def("__repr__", [](const ccScalarField::Range &rg) {
		std::string info = fmt::format(
			"Range with ({}, {}) in ({}, {})", rg.start(), rg.stop(), rg.min(), rg.max());
		return info;
	})
	.def("min",			&ccScalarField::Range::min, "Returns the minimum value")
	.def("max",			&ccScalarField::Range::max, "Returns the maximum value")
	.def("start",		&ccScalarField::Range::start, "Returns the current start value (in [min,max])")
	.def("stop",		&ccScalarField::Range::stop, "Returns the current stop value (in [min,max])")
	.def("range",		&ccScalarField::Range::range, "Returns the actual range: start-stop (but can't be ZERO!)")
	.def("max_range",	&ccScalarField::Range::maxRange, "Returns the maximum range")
	.def("set_bounds",	&ccScalarField::Range::setBounds, "Sets the bounds", "min_val"_a, "max_val"_a, "reset_start_stop"_a = true)
	.def("set_start",	&ccScalarField::Range::setStart, "Sets the current start value", "value"_a)
	.def("set_stop",	&ccScalarField::Range::setStop, "Sets the current stop value", "value"_a)
	.def("in_bound",	&ccScalarField::Range::inbound, "Returns the nearest inbound value", "value"_a)
	.def("is_in_bound", &ccScalarField::Range::isInbound, "Returns whether a value is inbound or not", "value"_a)
	.def("is_in_range", &ccScalarField::Range::isInRange, "Returns whether a value is inside range or not", "value"_a);
	docstring::ClassMethodDocInject(m, "Range", "min");
	docstring::ClassMethodDocInject(m, "Range", "max");
	docstring::ClassMethodDocInject(m, "Range", "start");
	docstring::ClassMethodDocInject(m, "Range", "stop");
	docstring::ClassMethodDocInject(m, "Range", "range");
	docstring::ClassMethodDocInject(m, "Range", "max_range");
	docstring::ClassMethodDocInject(m, "Range", "set_bounds");
	docstring::ClassMethodDocInject(m, "Range", "set_start");
	docstring::ClassMethodDocInject(m, "Range", "set_stop");
	docstring::ClassMethodDocInject(m, "Range", "in_bound");
	docstring::ClassMethodDocInject(m, "Range", "is_in_bound");
	docstring::ClassMethodDocInject(m, "Range", "is_in_range");

	// cloudViewer.utility.ccScalarField
	py::class_<ccScalarField, std::unique_ptr<ccScalarField, py::nodelete>, CVLib::ScalarField>
		scalarfield(m, "ccScalarField", "A scalar field associated to display-related parameters.");
	//py::detail::bind_copy_functions<ccScalarField>(scalarfield);
	scalarfield.def(py::init([](const std::string& name) {
		return new ccScalarField(name.c_str());
	}), "Simplified constructor", "name"_a = "ScalarField")
	.def("__repr__", [](const ccScalarField &sf) {
		std::string info = fmt::format(
			"ScalarField ({}) with {} scalars and range ({}, {}) ",
			sf.getName(), sf.currentSize(), sf.getMin(), sf.getMax());
		return info;
	})
	.def("display_range", &ccScalarField::displayRange, "Access to the range of displayed values.")
	.def("saturation_range", &ccScalarField::saturationRange, "Access to the range of saturation values.")
	.def("log_saturation_range", &ccScalarField::logSaturationRange, 
		"Access to the range of log scale saturation values.")
	.def("set_min_displayed", &ccScalarField::setMinDisplayed, 
		"Sets the minimum displayed value.", "value"_a)
	.def("set_max_displayed", &ccScalarField::setMaxDisplayed, 
		"Sets the maximum displayed value.", "value"_a)
	.def("set_saturation_start", &ccScalarField::setSaturationStart, 
		"Sets the value at which to start color gradient.", "value"_a)
	.def("set_saturation_stop", &ccScalarField::setSaturationStop, 
		"Sets the value at which to stop color gradient.", "value"_a)
	.def("get_color", [](const ccScalarField& sf, ScalarType value) {
		return ecvColor::Rgb::ToEigen(*sf.getColor(value));
	}, "Returns the color corresponding to a given value (wrt to the current display parameters).", "value"_a)
	.def("get_color_by_index", [](const ccScalarField& sf, unsigned index) {
		return ecvColor::Rgb::ToEigen(*sf.getValueColor(index));
	}, "Shortcut to getColor.", "index"_a)
	.def("show_nan_in_grey", &ccScalarField::showNaNValuesInGrey, 
		"Sets whether NaN/out of displayed range values should be displayed in gray or hidden.")
	.def("are_nan_shown_in_grey", &ccScalarField::areNaNValuesShownInGrey,
		"Returns whether NaN values are displayed in gray or hidden.")
	.def("always_show_zero", &ccScalarField::alwaysShowZero,
		"Sets whether 0 should always appear in associated color ramp or not.")
	.def("is_zero_always_shown", &ccScalarField::isZeroAlwaysShown,
		"Returns whether 0 should always appear in associated color ramp or not.")
	.def("set_symmetrical_scale", &ccScalarField::setSymmetricalScale,
		"Sets whether the color scale should be symmetrical or not.")
	.def("symmetrical_scale", &ccScalarField::symmetricalScale,
		"Returns whether the color scale s symmetrical or not.")
	.def("set_log_scale", &ccScalarField::setLogScale,
		"Sets whether scale is logarithmic or not.")
	.def("log_scale", &ccScalarField::logScale,
		"Returns whether scalar field is logarithmic or not.")
	.def("get_color_ramp_steps", &ccScalarField::getColorRampSteps,
		"Returns number of color ramp steps.")
	.def("set_color_ramp_steps", &ccScalarField::setColorRampSteps,
		"Sets number of color ramp steps used for display.")
	.def("may_have_hidden_values", &ccScalarField::mayHaveHiddenValues,
		"Returns whether the scalar field in its current configuration MAY have 'hidden' values or not.")
	.def("set_modification_flag", &ccScalarField::setModificationFlag, "Sets modification flag state.", "state"_a)
	.def("get_modification_flag", &ccScalarField::getModificationFlag, "Returns modification flag state.")
	.def("get_global_shift", &ccScalarField::getGlobalShift, "Returns the global shift (if any).")
	.def("set_global_shift", &ccScalarField::setGlobalShift, "Sets the global shift.")
	.def("is_serializable", &ccScalarField::isSerializable, "Returns whether object is serializable of not.")
	.def("to_file", [](const ccScalarField& sf, const std::string& filename) {
		QFile out(filename.c_str());
		if (!out.open(QIODevice::WriteOnly))
		{
			return false;
		}
		return sf.toFile(out);
	}, "Saves data to binary stream", "filename"_a)
	.def("from_file", [](ccScalarField& sf, const std::string& filename,
						short data_version, int flags) {
		QFile in(filename.c_str());
		if (!in.open(QIODevice::ReadOnly))
		{
			return false;
		}
		return sf.fromFile(in, data_version, flags);
	}, "Loads data from binary stream", "filename"_a, "data_version"_a, "flags"_a)
	.def("import_parameters_from", [](ccScalarField& sf, const ccScalarField& source) {
		sf.importParametersFrom(&source);
	}, "Imports the parameters from another scalar field", "source"_a);

	docstring::ClassMethodDocInject(m, "ccScalarField", "display_range");
	docstring::ClassMethodDocInject(m, "ccScalarField", "saturation_range");
	docstring::ClassMethodDocInject(m, "ccScalarField", "log_saturation_range");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_min_displayed");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_max_displayed");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_saturation_start");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_saturation_stop");
	docstring::ClassMethodDocInject(m, "ccScalarField", "get_color");
	docstring::ClassMethodDocInject(m, "ccScalarField", "get_color_by_index");
	docstring::ClassMethodDocInject(m, "ccScalarField", "show_nan_in_grey");
	docstring::ClassMethodDocInject(m, "ccScalarField", "are_nan_shown_in_grey");
	docstring::ClassMethodDocInject(m, "ccScalarField", "always_show_zero");
	docstring::ClassMethodDocInject(m, "ccScalarField", "is_zero_always_shown");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_symmetrical_scale");
	docstring::ClassMethodDocInject(m, "ccScalarField", "symmetrical_scale");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_log_scale");
	docstring::ClassMethodDocInject(m, "ccScalarField", "log_scale");
	docstring::ClassMethodDocInject(m, "ccScalarField", "get_color_ramp_steps");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_color_ramp_steps");
	docstring::ClassMethodDocInject(m, "ccScalarField", "may_have_hidden_values");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_modification_flag");
	docstring::ClassMethodDocInject(m, "ccScalarField", "get_modification_flag");
	docstring::ClassMethodDocInject(m, "ccScalarField", "get_global_shift");
	docstring::ClassMethodDocInject(m, "ccScalarField", "set_global_shift");
	docstring::ClassMethodDocInject(m, "ccScalarField", "is_serializable");
	docstring::ClassMethodDocInject(m, "ccScalarField", "to_file");
	docstring::ClassMethodDocInject(m, "ccScalarField", "from_file");
	docstring::ClassMethodDocInject(m, "ccScalarField", "import_parameters_from");
}