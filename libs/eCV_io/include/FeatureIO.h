// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_FEATURE_IO_HEADER
#define ECV_FEATURE_IO_HEADER

#include "eCV_io.h"

#include <ecvFeature.h>

namespace cloudViewer {
namespace io {

/// The general entrance for reading a Feature from a file
/// \return If the read function is successful.
bool ECV_IO_LIB_API ReadFeature(const std::string &filename, utility::Feature &feature);

/// The general entrance for writing a Feature to a file
/// \return If the write function is successful.
bool ECV_IO_LIB_API WriteFeature(const std::string &filename,
                                 const utility::Feature &feature);

bool ECV_IO_LIB_API ReadFeatureFromBIN(const std::string &filename,
                                       utility::Feature &feature);

bool ECV_IO_LIB_API WriteFeatureToBIN(const std::string &filename,
                                      const utility::Feature &feature);

}  // namespace io
}  // namespace cloudViewer

#endif // ECV_FEATURE_IO_HEADER
