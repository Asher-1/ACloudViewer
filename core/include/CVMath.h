// SPDX-License-Identifier: LGPL-2.0-or-later
// Copyright © ACloudViewer Project

#pragma once

//! \file CVMath.h
//! Methods for conversion and comparison. Note that these are intenionally not
//! templates. They are short methods and templates are overkill for these
//! cases.

#include "CVConst.h"

namespace cloudViewer {
//! Test a floating point number against our epsilon (a very small number).
/*!
  \param x The number to test
  \return True if the number is less than epsilon.
*/
inline bool LessThanEpsilon(float x) { return x < ZERO_TOLERANCE_F; }

//! Test a floating point number against our epsilon (a very small number).
/*!
  \param x The number to test
  \return True if the number is less than epsilon.
*/
inline bool LessThanEpsilon(double x) { return x < ZERO_TOLERANCE_D; }

//! Test a floating point number against our epsilon (a very small number).
/*!
  \param x The number to test
  \return True if the number is greater than epsilon.
*/
inline bool GreaterThanEpsilon(float x) { return x > ZERO_TOLERANCE_F; }

//! Test a floating point number against our epsilon (a very small number).
/*!
  \param x The number to test
  \return True if the number is greater than epsilon.
*/
inline bool GreaterThanEpsilon(double x) { return x > ZERO_TOLERANCE_D; }

//! Test a (squared) floating point number against our epsilon (a very small
//! number).
/*!
  \param x The number to test
  \return True if the number is greater than epsilon.
*/
inline bool GreaterThanSquareEpsilon(double x) {
    return x > ZERO_SQUARED_TOLERANCE_D;
}

//! Test a (squared) floating point number against our epsilon (a very small
//! number).
/*!
  \param x The (squared) number to test
  \return True if the number is less than epsilon.
*/
inline bool LessThanSquareEpsilon(double x) {
    return x < ZERO_SQUARED_TOLERANCE_D;
}

//! Convert radians to degrees.
/*!
  \param radians Radians to convert
  \return \a radians represented as degrees.
*/
inline float RadiansToDegrees(int radians) {
    return static_cast<float>(radians) * (180.0f / static_cast<float>(M_PI));
}

//! Convert radians to degrees.
/*!
  \param radians Radians to convert
  \return \a radians represented as degrees.
*/
inline float RadiansToDegrees(float radians) {
    return radians * (180.0f / static_cast<float>(M_PI));
}

//! Convert radians to degrees.
/*!
  \param radians Radians to convert
  \return \a radians represented as degrees.
*/
inline double RadiansToDegrees(double radians) {
    return radians * (180.0 / M_PI);
}

//! Convert degrees to radians.
/*!
  \param degrees Degrees to convert
  \return \a degrees represented as radians.
*/
inline float DegreesToRadians(int degrees) {
    return static_cast<float>(degrees) * (static_cast<float>(M_PI) / 180.0f);
}

//! Convert degrees to radians.
/*!
  \param degrees Degrees to convert
  \return \a degrees represented as radians.
*/
inline float DegreesToRadians(float degrees) {
    return degrees * (static_cast<float>(M_PI) / 180.0f);
}

//! Convert degrees to radians.
/*!
  \param degrees Degrees to convert
  \return \a degrees represented as radians.
*/
inline double DegreesToRadians(double degrees) {
    return degrees * (M_PI / 180.0);
}
}  // namespace cloudViewer
