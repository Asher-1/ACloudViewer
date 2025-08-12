// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// PRIVATE RealSense header for compiling Open3D. Do not #include outside
// Open3D.
#pragma once

#include <librealsense2/rs.hpp>

#include <IJsonConvertibleIO.h>

namespace cloudViewer {
namespace t {
namespace io {

DECLARE_STRINGIFY_ENUM(rs2_stream)
DECLARE_STRINGIFY_ENUM(rs2_format)
DECLARE_STRINGIFY_ENUM(rs2_l500_visual_preset)
DECLARE_STRINGIFY_ENUM(rs2_rs400_visual_preset)
DECLARE_STRINGIFY_ENUM(rs2_sr300_visual_preset)

}  // namespace io
}  // namespace t
}  // namespace cloudViewer
