// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file ecvTools.h
 * @brief Utility functions for vector ops and color conversion (non-PCL).
 */

#ifdef _MSC_VER
#pragma warning(disable : 4996)
#pragma warning(disable : 4819)
#endif

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>

// CV_DB_LIB
#include <ecvColorTypes.h>

namespace ecvTools {
/// @param a First sorted vector.
/// @param b Second sorted vector.
/// @return Intersection of a and b.
static std::vector<int> IntersectionVector(std::vector<int>& a,
                                           std::vector<int>& b) {
    std::vector<int> c;
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    set_intersection(a.begin(), a.end(), b.begin(), b.end(), back_inserter(c));
    return c;
}

static std::vector<int> UnionVector(std::vector<int>& a, std::vector<int>& b) {
    std::vector<int> c;
    set_union(a.begin(), a.end(), b.begin(), b.end(), back_inserter(c));
    return c;
}

/// @param a First vector.
/// @param b Second vector.
/// @return Elements in a but not in b.
static std::vector<int> DiffVector(std::vector<int> a, std::vector<int> b) {
    std::vector<int> c;
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    set_difference(a.begin(), a.end(), b.begin(), b.end(), back_inserter(c));
    return c;
}

static ecvColor::Rgbf TransFormRGB(const ecvColor::Rgb& col) {
    if (col.r <= 1 && col.g <= 1 && col.b <= 1) {
        return ecvColor::Rgbf(col.r, col.g, col.b);
    } else {
        return ecvColor::Rgbf(col.r / 255.0f, col.g / 255.0f, col.b / 255.0f);
    }
}

static ecvColor::Rgbf TransFormRGB(const ecvColor::Rgbf& col) {
    if (col.r <= 1 && col.g <= 1 && col.b <= 1) {
        return ecvColor::Rgbf(col.r, col.g, col.b);
    } else {
        return ecvColor::Rgbf(col.r / 255.0f, col.g / 255.0f, col.b / 255.0f);
    }
}

};  // namespace ecvTools
