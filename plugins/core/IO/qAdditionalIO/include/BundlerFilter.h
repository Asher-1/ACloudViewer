// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FileIOFilter.h"

//! Noah Snavely's Bundler output file filter
/** See http://phototour.cs.washington.edu/
 **/
class BundlerFilter : public FileIOFilter {
public:
    BundlerFilter();

    // inherited from FileIOFilter
    virtual CC_FILE_ERROR loadFile(const QString& filename,
                                   ccHObject& container,
                                   LoadParameters& parameters) override;

    //! Specific load method
    CC_FILE_ERROR loadFileExtended(
            const QString& filename,
            ccHObject& container,
            LoadParameters& parameters,
            const QString& altKeypointsFilename = QString(),
            bool undistortImages = false,
            bool generateColoredDTM = false,
            unsigned coloredDTMVerticesCount = 1000000,
            float scaleFactor = 1.0f);
};
