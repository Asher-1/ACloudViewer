// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file CVMath.h
 * @brief Mathematical utility functions for conversions and comparisons
 *
 * Provides methods for angle conversion and floating-point comparisons.
 * Note: These are intentionally not templates - they are short methods
 * and templates would be overkill for these cases.
 */

#include "CVConst.h"

namespace cloudViewer {

/**
 * @brief Test if a float is less than epsilon
 * @param x The number to test
 * @return true if the number is less than epsilon
 */
inline bool LessThanEpsilon(float x) { return x < ZERO_TOLERANCE_F; }

/**
 * @brief Test if a double is less than epsilon
 * @param x The number to test
 * @return true if the number is less than epsilon
 */
inline bool LessThanEpsilon(double x) { return x < ZERO_TOLERANCE_D; }

/**
 * @brief Test if a float is greater than epsilon
 * @param x The number to test
 * @return true if the number is greater than epsilon
 */
inline bool GreaterThanEpsilon(float x) { return x > ZERO_TOLERANCE_F; }

/**
 * @brief Test if a double is greater than epsilon
 * @param x The number to test
 * @return true if the number is greater than epsilon
 */
inline bool GreaterThanEpsilon(double x) { return x > ZERO_TOLERANCE_D; }

/**
 * @brief Test if a squared double is greater than epsilon
 * @param x The squared number to test
 * @return true if the number is greater than epsilon squared
 */
inline bool GreaterThanSquareEpsilon(double x) {
    return x > ZERO_SQUARED_TOLERANCE_D;
}

/**
 * @brief Test if a squared double is less than epsilon
 * @param x The squared number to test
 * @return true if the number is less than epsilon squared
 */
inline bool LessThanSquareEpsilon(double x) {
    return x < ZERO_SQUARED_TOLERANCE_D;
}

/**
 * @brief Convert radians to degrees (int overload)
 * @param radians Angle in radians
 * @return Angle in degrees
 */
inline float RadiansToDegrees(int radians) {
    return static_cast<float>(radians) * (180.0f / static_cast<float>(M_PI));
}

/**
 * @brief Convert radians to degrees (float overload)
 * @param radians Angle in radians
 * @return Angle in degrees
 */
inline float RadiansToDegrees(float radians) {
    return radians * (180.0f / static_cast<float>(M_PI));
}

/**
 * @brief Convert radians to degrees (double overload)
 * @param radians Angle in radians
 * @return Angle in degrees
 */
inline double RadiansToDegrees(double radians) {
    return radians * (180.0 / M_PI);
}

/**
 * @brief Convert degrees to radians (int overload)
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
inline float DegreesToRadians(int degrees) {
    return static_cast<float>(degrees) * (static_cast<float>(M_PI) / 180.0f);
}

/**
 * @brief Convert degrees to radians (float overload)
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
inline float DegreesToRadians(float degrees) {
    return degrees * (static_cast<float>(M_PI) / 180.0f);
}

/**
 * @brief Convert degrees to radians (double overload)
 * @param degrees Angle in degrees
 * @return Angle in radians
 */
inline double DegreesToRadians(double degrees) {
    return degrees * (M_PI / 180.0);
}
}  // namespace cloudViewer
