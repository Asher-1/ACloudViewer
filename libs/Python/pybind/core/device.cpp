// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/Device.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace core {

void pybind_core_device(py::module &m) {
    py::class_<Device> device(
            m, "Device",
            "Device context specifying device type and device id.");
            device.def(py::init<>());
            device.def(py::init<Device::DeviceType, int>());
            device.def(py::init<const std::string &, int>());
            device.def(py::init<const std::string &>());
            device.def("__eq__", &Device::operator==);
            device.def("__ene__", &Device::operator!=);
            device.def("__repr__", [](const Device &d) {
                std::string device_type;
                switch (d.GetType()) {
                    case Device::DeviceType::CPU:
                        device_type = "CPU";
                        break;
                    case Device::DeviceType::CUDA:
                        device_type = "CUDA";
                        break;
                    case Device::DeviceType::SYCL:
                        device_type = "SYCL";
                        break;
                    default:
                        utility::LogError("Unknown device type");
                        return d.ToString();
                }
                return fmt::format("Device(\"{}\", {})", device_type, d.GetID());
            });
            device.def("__str__", &Device::ToString);
            device.def("get_type", &Device::GetType);
            device.def("get_id", &Device::GetID);
            device.def(py::pickle(
                    [](const Device &d) {
                        return py::make_tuple(d.GetType(), d.GetID());
                    },
                    [](py::tuple t) {
                        if (t.size() != 2) {
                            utility::LogError(
                                    "Cannot unpickle Device! Expecting a tuple of size "
                                    "2.");
                        }
                        return Device(t[0].cast<Device::DeviceType>(),
                                      t[1].cast<int>());
                    }));

    py::native_enum<Device::DeviceType>(device, "DeviceType", "enum.Enum")
            .value("CPU", Device::DeviceType::CPU)
            .value("CUDA", Device::DeviceType::CUDA)
            .value("SYCL", Device::DeviceType::SYCL)
            .finalize();
}

}  // namespace core
}  // namespace cloudViewer
