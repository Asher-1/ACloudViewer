// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/reconstruction/sensor/sensor.h"

#include <cstring>
#include <unordered_map>

#include "pybind/docstring.h"
#include "sensor/bitmap.h"
#include "util/logging.h"

namespace cloudViewer {
namespace reconstruction {
namespace sensor {

// Upstream pycolmap parity (src/pycolmap/sensor/bitmap.cc): the Bitmap value
// type as plain Open3D-style class bindings; the numpy views are materialized
// copies of the packed row-major pixel storage.
using colmap::Bitmap;
using colmap::BitmapColor;
using colmap::BitmapFormat;
using colmap::BitmapRescaleFilter;

static const std::unordered_map<std::string, std::string>
        map_shared_argument_docstrings = {
                {"path", "Path to the image file."},
                {"as_rgb",
                 "Whether to load the image in RGB (grayscale "
                 "otherwise)."},
};

void pybind_sensor(py::module& m) {
    py::module m_sensor = m.def_submodule("sensor");

    py::enum_<BitmapFormat>(m_sensor, "BitmapFormat",
                            "Image file format selectors.")
            .value("UNKNOWN", BitmapFormat::kUnknown)
            .value("PNG", BitmapFormat::kPng)
            .value("JPEG", BitmapFormat::kJpeg)
            .value("TIFF", BitmapFormat::kTiff);

    py::enum_<BitmapRescaleFilter>(m_sensor, "BitmapRescaleFilter",
                                   "Rescaling filters.")
            .value("BILINEAR", BitmapRescaleFilter::kBilinear)
            .value("BOX", BitmapRescaleFilter::kBox);

    py::class_<BitmapColor<uint8_t>> bitmap_color(m_sensor, "BitmapColor",
                                                  "An 8-bit RGB pixel color.");
    bitmap_color.def(py::init<>())
            .def(py::init<uint8_t>(), "gray"_a)
            .def(py::init<uint8_t, uint8_t, uint8_t>(), "r"_a, "g"_a, "b"_a)
            .def_readwrite("r", &BitmapColor<uint8_t>::r)
            .def_readwrite("g", &BitmapColor<uint8_t>::g)
            .def_readwrite("b", &BitmapColor<uint8_t>::b)
            .def("__repr__", [](const BitmapColor<uint8_t>& self) {
                return "BitmapColor(r=" + std::to_string(self.r) +
                       ", g=" + std::to_string(self.g) +
                       ", b=" + std::to_string(self.b) + ")";
            });

    py::class_<Bitmap> bitmap(m_sensor, "Bitmap",
                              "An 8-bit grayscale or RGB image with EXIF "
                              "metadata access.");
    bitmap.def(py::init<>())
            .def(py::init([](int width, int height, bool as_rgb) {
                     Bitmap bmp;
                     if (!bmp.Allocate(width, height, as_rgb)) {
                         throw std::runtime_error(
                                 "Failed to allocate the bitmap");
                     }
                     return bmp;
                 }),
                 "width"_a, "height"_a, "as_rgb"_a = true)
            .def(
                    "allocate",
                    [](Bitmap& self, int width, int height, bool as_rgb) {
                        return self.Allocate(width, height, as_rgb);
                    },
                    "width"_a, "height"_a, "as_rgb"_a = true,
                    "Allocate the pixel storage (contents undefined).")
            .def_property_readonly("width", &Bitmap::Width)
            .def_property_readonly("height", &Bitmap::Height)
            .def_property_readonly("channels", &Bitmap::Channels)
            .def_property_readonly("is_rgb", &Bitmap::IsRGB)
            .def_property_readonly("is_grey", &Bitmap::IsGrey)
            .def_property_readonly("num_bytes", &Bitmap::NumBytes)
            .def(
                    "read",
                    [](Bitmap& self, const std::string& path, bool as_rgb) {
                        return self.Read(path, as_rgb);
                    },
                    "path"_a, "as_rgb"_a = true,
                    "Read the image from the given path.")
            .def(
                    "write",
                    [](const Bitmap& self, const std::string& path,
                       BitmapFormat format,
                       int flags) { return self.Write(path, format, flags); },
                    "path"_a, "format"_a = BitmapFormat::kUnknown,
                    "flags"_a = 0,
                    "Write the image to the given path (flags is the JPEG "
                    "quality [1, 100]).")
            .def(
                    "array",
                    [](const Bitmap& self) {
                        THROW_CHECK(self.IsGrey() || self.IsRGB());
                        const int channels = self.Channels();
                        py::array_t<uint8_t> arr(
                                {static_cast<py::ssize_t>(self.Height()),
                                 static_cast<py::ssize_t>(self.Width()),
                                 static_cast<py::ssize_t>(channels)});
                        // Bitmap::Data() exposes the Storage object, not the
                        // pixel buffer - materialize via the row-major copy.
                        const auto pixels = self.ConvertToRawBits();
                        THROW_CHECK_EQ(pixels.size(), self.NumBytes());
                        std::memcpy(arr.mutable_data(), pixels.data(),
                                    pixels.size());
                        return arr;
                    },
                    "A (height, width, channels) uint8 copy of the pixels.")
            .def(
                    "set_array",
                    [](Bitmap& self,
                       const py::array_t<uint8_t, py::array::c_style |
                                                          py::array::forcecast>&
                               arr) {
                        const int channels = self.Channels();
                        if (channels == 1) {
                            THROW_CHECK_EQ(arr.ndim(), 2);
                        } else {
                            THROW_CHECK_EQ(arr.ndim(), 3);
                            THROW_CHECK_EQ(static_cast<int>(arr.shape(2)),
                                           channels);
                        }
                        THROW_CHECK_EQ(static_cast<int>(arr.shape(0)),
                                       self.Height());
                        THROW_CHECK_EQ(static_cast<int>(arr.shape(1)),
                                       self.Width());
                        // There is no bulk pixel-write surface on the fork's
                        // Bitmap (Data() is the Storage handle), so write via
                        // SetPixel. O(width*height) by design.
                        auto buf = arr.unchecked();
                        for (py::ssize_t y = 0; y < buf.shape(0); ++y) {
                            for (py::ssize_t x = 0; x < buf.shape(1); ++x) {
                                if (channels == 1) {
                                    self.SetPixel(
                                            static_cast<int>(x),
                                            static_cast<int>(y),
                                            BitmapColor<uint8_t>(buf(y, x)));
                                } else {
                                    self.SetPixel(
                                            static_cast<int>(x),
                                            static_cast<int>(y),
                                            BitmapColor<uint8_t>(buf(y, x, 0),
                                                                 buf(y, x, 1),
                                                                 buf(y, x, 2)));
                                }
                            }
                        }
                    },
                    "array"_a,
                    "Copy a (height, width[, channels]) uint8 array into the "
                    "bitmap.")
            .def(
                    "get_pixel",
                    [](const Bitmap& self, int x, int y) -> py::object {
                        const auto color = self.GetPixel(x, y);
                        if (!color) {
                            return py::none();
                        }
                        return py::cast(*color);
                    },
                    "x"_a, "y"_a,
                    "The pixel color at the given coordinates or None.")
            .def(
                    "set_pixel",
                    [](Bitmap& self, int x, int y,
                       const BitmapColor<uint8_t>& color) {
                        return self.SetPixel(x, y, color);
                    },
                    "x"_a, "y"_a, "color"_a,
                    "Set the pixel color at the given coordinates.")
            .def(
                    "fill",
                    [](Bitmap& self, const BitmapColor<uint8_t>& color) {
                        self.Fill(color);
                    },
                    "color"_a, "Fill the image with a constant color.")
            .def("smooth", &Bitmap::Smooth, "sigma_x"_a, "sigma_y"_a,
                 "Smooth the image with a Gaussian kernel.")
            .def("rescale", &Bitmap::Rescale, "new_width"_a, "new_height"_a,
                 "filter"_a = BitmapRescaleFilter::kBilinear,
                 "Rescale the image to the new dimensions.")
            .def("clone", &Bitmap::Clone, "Return a copy of the image.")
            .def("clone_as_grey", &Bitmap::CloneAsGrey,
                 "Return a grayscale copy of the image.")
            .def("clone_as_rgb", &Bitmap::CloneAsRGB,
                 "Return an RGB copy of the image.")
            .def(
                    "exif_focal_length",
                    [](const Bitmap& self) -> py::object {
                        double focal_length = -1.0;
                        if (!self.ExifFocalLength(&focal_length)) {
                            return py::none();
                        }
                        return py::cast(focal_length);
                    },
                    "The EXIF focal length in pixels or None.")
            .def(
                    "exif_latitude",
                    [](const Bitmap& self) -> py::object {
                        double value = 0.0;
                        if (!self.ExifLatitude(&value)) {
                            return py::none();
                        }
                        return py::cast(value);
                    },
                    "The EXIF GPS latitude or None.")
            .def(
                    "exif_longitude",
                    [](const Bitmap& self) -> py::object {
                        double value = 0.0;
                        if (!self.ExifLongitude(&value)) {
                            return py::none();
                        }
                        return py::cast(value);
                    },
                    "The EXIF GPS longitude or None.")
            .def(
                    "exif_altitude",
                    [](const Bitmap& self) -> py::object {
                        double value = 0.0;
                        if (!self.ExifAltitude(&value)) {
                            return py::none();
                        }
                        return py::cast(value);
                    },
                    "The EXIF GPS altitude or None.")
            .def("exif_orientation", &Bitmap::ExifOrientation,
                 "The EXIF orientation tag or None.")
            .def(
                    "exif_camera_model",
                    [](const Bitmap& self) -> py::object {
                        std::string model;
                        if (!self.ExifCameraModel(&model)) {
                            return py::none();
                        }
                        return py::cast(model);
                    },
                    "The EXIF camera model string or None.")
            .def("__repr__", [](const Bitmap& self) {
                return "Bitmap(width=" + std::to_string(self.Width()) +
                       ", height=" + std::to_string(self.Height()) +
                       ", channels=" + std::to_string(self.Channels()) + ")";
            });

    m_sensor.attr("__docstring__") = map_shared_argument_docstrings;
}

}  // namespace sensor
}  // namespace reconstruction
}  // namespace cloudViewer
