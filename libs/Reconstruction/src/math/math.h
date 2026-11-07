// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <list>
#include <stdexcept>
#include <vector>

#include "util/logging.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

namespace colmap {

// Return 1 if number is positive (including 0), -1 if negative.
// Undefined behavior if the value is NaN.
// Upstream parity (a2bd15c2 math/math.h): the homography decomposition
// multiplies square roots by these signs, so a zero sign would erase a term
// whenever a minor rounds to exactly 0.
template <typename T>
int SignOfNumber(const T val);

// Check if the given floating point number is a not-a-number (NaN) value.
inline bool IsNaN(const float x);
inline bool IsNaN(const double x);

// Check if the given floating point number is a infinity.
inline bool IsInf(const float x);
inline bool IsInf(const double x);

// Clip the given value to a low and maximum value.
template <typename T>
inline T Clip(const T& value, const T& low, const T& high);

// Convert angle in degree to radians.
inline float DegToRad(const float deg);
inline double DegToRad(const double deg);

// Convert angle in radians to degree.
inline float RadToDeg(const float rad);
inline double RadToDeg(const double rad);

// Determine median value in vector. Returns NaN for empty vectors.
template <typename T>
double Median(const std::vector<T>& elems);

// Determine mean value in a vector.
template <typename T>
double Mean(const std::vector<T>& elems);

// Determine sample variance in a vector.
template <typename T>
double Variance(const std::vector<T>& elems);

// Determine sample standard deviation in a vector.
template <typename T>
double StdDev(const std::vector<T>& elems);

// Check if any of the values in the vector is less than the given threshold.
template <typename T>
bool AnyLessThan(std::vector<T> elems, T threshold);

// Check if any of the values in the vector is greater than the given threshold.
template <typename T>
bool AnyGreaterThan(std::vector<T> elems, T threshold);

// Generate N-choose-K combinations.
//
// Note that elements in range [first, last) must be in sorted order,
// according to `std::less`.
template <class Iterator>
bool NextCombination(Iterator first, Iterator middle, Iterator last);

// Sigmoid function.
template <typename T>
T Sigmoid(const T x, const T alpha = 1);

// Scale values according to sigmoid transform.
//
//   x \in [0, 1] -> x \in [-x0, x0] -> sigmoid(x, alpha) -> x \in [0, 1]
//
// @param x        Value to be scaled in the range [0, 1].
// @param x0       Spread that determines the range x is scaled to.
// @param alpha    Exponential sigmoid factor.
//
// @return         The scaled value in the range [0, 1].
template <typename T>
T ScaleSigmoid(T x, const T alpha = 1, const T x0 = 10);

// Binomial coefficient or all combinations, defined as n! / ((n - k)! k!).
size_t NChooseK(const size_t n, const size_t k);

// Cast value from one type to another and truncate instead of overflow, if the
// input value is out of range of the output data type.
template <typename T1, typename T2>
T2 TruncateCast(const T1 value);

// Compute the n-th percentile in the given sequence. Performs linear
// interpolation between the closest ranks.
template <typename T>
double Percentile(const std::vector<T>& elems, const double p);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template <class Iterator>
bool NextCombination(Iterator first1,
                     Iterator last1,
                     Iterator first2,
                     Iterator last2) {
    if ((first1 == last1) || (first2 == last2)) {
        return false;
    }
    Iterator m1 = last1;
    Iterator m2 = last2;
    --m2;
    while (--m1 != first1 && *m1 >= *m2) {
    }
    bool result = (m1 == first1) && *first1 >= *m2;
    if (!result) {
        while (first2 != m2 && *m1 >= *first2) {
            ++first2;
        }
        first1 = m1;
        std::iter_swap(first1, first2);
        ++first1;
        ++first2;
    }
    if ((first1 != last1) && (first2 != last2)) {
        m1 = last1;
        m2 = first2;
        while ((m1 != first1) && (m2 != last2)) {
            std::iter_swap(--m1, m2);
            ++m2;
        }
        std::reverse(first1, m1);
        std::reverse(first1, last1);
        std::reverse(m2, last2);
        std::reverse(first2, last2);
    }
    return !result;
}

}  // namespace internal

template <typename T>
int SignOfNumber(const T val) {
    return val >= 0 ? 1 : -1;
}

// Clamp the given value to a low and maximum value (upstream parity,
// dbb41680 math/math.h).
template <typename T>
inline T Clamp(const T& value, const T& low, const T& high) {
    return std::max(low, std::min(value, high));
}

bool IsNaN(const float x) { return x != x; }
bool IsNaN(const double x) { return x != x; }

bool IsInf(const float x) { return !IsNaN(x) && IsNaN(x - x); }
bool IsInf(const double x) { return !IsNaN(x) && IsNaN(x - x); }

template <typename T>
T Clip(const T& value, const T& low, const T& high) {
    return std::max(low, std::min(value, high));
}

float DegToRad(const float deg) {
    return deg * 0.0174532925199432954743716805978692718781530857086181640625f;
}

double DegToRad(const double deg) {
    return deg * 0.0174532925199432954743716805978692718781530857086181640625;
}

// Convert angle in radians to degree.
float RadToDeg(const float rad) {
    return rad * 57.29577951308232286464772187173366546630859375f;
}

double RadToDeg(const double rad) {
    return rad * 57.29577951308232286464772187173366546630859375;
}

template <typename T>
double Median(const std::vector<T>& elems) {
    // Upstream parity (b2cdd169 #3041): the median is the interpolated 50th
    // percentile, accumulated in double so integer inputs cannot overflow.
    return Percentile(elems, 50);
}

// Compute median absolute deviation (MAD) (upstream parity, dbb41680
// math/math.h): returns {median, MAD} of the elements.
template <typename T>
std::pair<double, double> MedianAbsoluteDeviation(std::vector<T>& elems) {
    const double median = Median(elems);
    std::vector<double> abs_deviations(elems.size());
    for (size_t i = 0; i < elems.size(); i++) {
        abs_deviations[i] = std::abs(static_cast<double>(elems[i]) - median);
    }
    return {median, Median(abs_deviations)};
}

template <typename T>
std::pair<double, double> MedianAbsoluteDeviation(std::vector<T>&& elems) {
    return MedianAbsoluteDeviation(elems);
}

template <typename T>
double Percentile(const std::vector<T>& elems, const double p) {
    CHECK(!elems.empty());
    CHECK_GE(p, 0);
    CHECK_LE(p, 100);

    // Upstream parity (b2cdd169 #3041): linear interpolation between the two
    // closest ranks instead of rounding to the nearest rank. The fork keeps
    // the const-reference signature, so it partitions a copy.
    const double idx_double = p / 100. * (elems.size() - 1);
    const double left_idx_double = std::floor(idx_double);
    const size_t left_idx = static_cast<size_t>(left_idx_double);
    const double right_idx_double = std::ceil(idx_double);
    const size_t right_idx = static_cast<size_t>(right_idx_double);

    std::vector<T> ordered_elems = elems;
    std::nth_element(ordered_elems.begin(), ordered_elems.begin() + right_idx,
                     ordered_elems.end());
    const double right = ordered_elems[right_idx];
    if (left_idx == right_idx) {
        return right;
    }
    const double left = *std::max_element(ordered_elems.begin(),
                                          ordered_elems.begin() + right_idx);
    return (right_idx_double - idx_double) * left +
           (idx_double - left_idx_double) * right;
}

template <typename T>
double Mean(const std::vector<T>& elems) {
    CHECK(!elems.empty());
    double sum = 0;
    for (const auto el : elems) {
        sum += static_cast<double>(el);
    }
    return sum / elems.size();
}

template <typename T>
double Variance(const std::vector<T>& elems) {
    const double mean = Mean(elems);
    double var = 0;
    for (const auto el : elems) {
        const double diff = el - mean;
        var += diff * diff;
    }
    return var / (elems.size() - 1);
}

template <typename T>
double StdDev(const std::vector<T>& elems) {
    return std::sqrt(Variance(elems));
}

template <typename T>
bool AnyLessThan(std::vector<T> elems, T threshold) {
    for (const auto& el : elems) {
        if (el < threshold) {
            return true;
        }
    }
    return false;
}

template <typename T>
bool AnyGreaterThan(std::vector<T> elems, T threshold) {
    for (const auto& el : elems) {
        if (el > threshold) {
            return true;
        }
    }
    return false;
}

template <class Iterator>
bool NextCombination(Iterator first, Iterator middle, Iterator last) {
    return internal::NextCombination(first, middle, middle, last);
}

template <typename T>
T Sigmoid(const T x, const T alpha) {
    return T(1) / (T(1) + exp(-x * alpha));
}

template <typename T>
T ScaleSigmoid(T x, const T alpha, const T x0) {
    const T t0 = Sigmoid(-x0, alpha);
    const T t1 = Sigmoid(x0, alpha);
    x = (Sigmoid(2 * x0 * x - x0, alpha) - t0) / (t1 - t0);
    return x;
}

template <typename T1, typename T2>
T2 TruncateCast(const T1 value) {
    using InputLimits = std::numeric_limits<T1>;
    using OutputLimits = std::numeric_limits<T2>;
    // Upstream parity (9e32193d math/math.h): the previous helper converted
    // the destination bounds to the source type, which wraps for out-of-range
    // values (e.g. TruncateCast<uint16_t, int16_t>(1234) returned -32768) and
    // used min() as the wrong lower bound for floating destinations.
    if constexpr (InputLimits::is_integer && OutputLimits::is_integer) {
        if constexpr (InputLimits::is_signed && !OutputLimits::is_signed) {
            if (value < 0) {
                return OutputLimits::lowest();
            }
        }
        if constexpr (InputLimits::digits > OutputLimits::digits) {
            if (value > static_cast<T1>(OutputLimits::max())) {
                return OutputLimits::max();
            }
            if constexpr (InputLimits::is_signed && OutputLimits::is_signed) {
                if (value < static_cast<T1>(OutputLimits::lowest())) {
                    return OutputLimits::lowest();
                }
            }
        }
    } else if constexpr (OutputLimits::is_integer &&
                         OutputLimits::digits <= InputLimits::digits) {
        // The integer bounds are exactly representable in the floating input
        // type.
        return static_cast<T2>(
                std::clamp(value, static_cast<T1>(OutputLimits::lowest()),
                           static_cast<T1>(OutputLimits::max())));
    } else {
        const long double input = static_cast<long double>(value);
        if (input <= static_cast<long double>(OutputLimits::lowest())) {
            return OutputLimits::lowest();
        }
        if (input >= static_cast<long double>(OutputLimits::max())) {
            return OutputLimits::max();
        }
    }
    return static_cast<T2>(value);
}

// 95% chi-square quantile for 3 degrees of freedom. Upstream COLMAP
// dbb41680 math/math.h parity; used by the robust model alignment (W10)
// and the GLomap stack (W4).
constexpr double kChiSquare95ThreeDof = 7.814727903251179;

}  // namespace colmap
