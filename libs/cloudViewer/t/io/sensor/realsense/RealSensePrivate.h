// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// PRIVATE RealSense header for compiling CloudViewer. Do not #include outside
// CloudViewer.
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
