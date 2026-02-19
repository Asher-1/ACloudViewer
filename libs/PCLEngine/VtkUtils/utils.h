// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file utils.h
 * @brief General utility functions for CloudViewer
 *
 * Provides miscellaneous helper functions including:
 * - Random number generation
 * - Color conversion (Qt ↔ VTK)
 * - Array operations (comparison, assignment, normalization)
 * - Math utilities
 * - File system helpers
 */

#pragma once

#include <QColor>
#include <QImage>
#include <QtMath>
#include <complex>

#include "qPCL.h"

class vtkActor;

/**
 * @namespace Utils
 * @brief General utility functions for CloudViewer
 *
 * Collection of helper functions for common operations.
 */
namespace Utils {

// =====================================================================
// String and UI Utilities
// =====================================================================

/**
 * @brief Get alphabetical character by index
 * @param index Character index (0=A, 1=B, etc.)
 * @return String containing the character
 */
QString QPCL_ENGINE_LIB_API character(int index);

/**
 * @brief Open file explorer at path
 * @param path Directory path to open
 *
 * Opens system file explorer/finder at specified location.
 */
void QPCL_ENGINE_LIB_API explorer(const QString& path);

/**
 * @brief Generate star icon image
 * @param size Image size (default: 30x30)
 * @return QImage containing star icon
 */
QImage QPCL_ENGINE_LIB_API star(const QSize& size = QSize(30, 30));

// =====================================================================
// Random Number Generation
// =====================================================================

/**
 * @brief Generate random double in range
 * @param low Lower bound
 * @param high Upper bound
 * @return Random value between low and high
 */
double QPCL_ENGINE_LIB_API random(int low, int high);

/**
 * @brief Generate random complex number
 * @tparam T Numeric type for complex components
 * @param low Lower bound for both real and imaginary parts
 * @param high Upper bound for both real and imaginary parts
 * @return Random complex number
 */
template <typename T>
inline static std::complex<T> random(int low, int high) {
    std::complex<T> c(random(low, high), random(low, high));
    return c;
}

// =====================================================================
// Color Conversion
// =====================================================================

/**
 * @brief Convert Qt color to VTK color
 * @param clr Qt color to convert
 * @param vtkClr Output VTK color array (double[3])
 *
 * Converts QColor to VTK's RGB format (0.0-1.0 range).
 */
void QPCL_ENGINE_LIB_API vtkColor(const QColor& clr, double* vtkClr);

/**
 * @brief Convert VTK color to Qt color
 * @param pClr VTK color array (double[3])
 * @return Qt QColor
 *
 * Converts VTK's RGB format to QColor.
 */
QColor QPCL_ENGINE_LIB_API qColor(double* pClr);

/**
 * @brief Convert Qt color to HSV values
 * @param clr Qt color
 * @param hsv Output HSV array (double[3])
 *
 * Converts QColor to HSV (Hue, Saturation, Value) format.
 */
void QPCL_ENGINE_LIB_API qColor2HSV(const QColor& clr, double* hsv);

template <typename T, int size = 3>
class ArrayComparator {
public:
    bool operator()(const T* lhs, const T* rhs) {
        for (auto i = 0; i < size; ++i) {
            if (lhs[i] != rhs[i]) return false;
        }
        return true;
    }
};

template <typename T, int size = 3>
class ArrayAssigner {
public:
    void operator()(T* lhs, const T* rhs) {
        // memset(lhs, rhs, 3 * sizeof(T));
        for (auto i = 0; i < size; ++i) lhs[i] = rhs[i];
    }
};

template <typename T, int size = 3>
class ArrayInitializer {
public:
    void operator()(T* array, T value = T()) {
        for (auto i = 0; i < size; ++i) array[i] = value;
    }
};

class Normalizer {
public:
    void operator()(const double* input, double* output) {
        double mod = qSqrt(input[0] * input[0] + input[1] * input[1] +
                           input[2] * input[2]);
        if (mod == 0) {
            output[0] = input[0];
            output[1] = input[1];
            output[2] = input[2];
        } else {
            output[0] = input[0] / mod;
            output[1] = input[1] / mod;
            output[2] = input[2] / mod;
        }
    }
};

typedef QList<vtkActor*> ActorList;

template <class T>
inline void vtkSafeDelete(T* obj) {
    if (obj) obj->Delete();
}

template <class T>
inline void vtkSafeDelete(QList<T*>& objList) {
    foreach (T* obj, objList) obj->Delete();
    objList.clear();
}

template <typename T>
inline T boundedValue(const T& value, const T& min, const T& max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

template <typename T>
static inline double module(T* vector) {
    return qSqrt(vector[0] * vector[0] + vector[1] * vector[1] +
                 vector[2] * vector[2]);
}

template <typename T>
static inline double distance(T* pot1, T* pot2) {
    double dX = pot2[0] - pot1[0];
    double dY = pot2[1] - pot1[1];
    double dZ = pot2[2] - pot1[2];
    return qSqrt(dX * dX + dY * dY + dZ * dZ);
}

/*!
 * \brief get vector between two points
 */
template <typename T>
static inline void normal(T* inPot1, T* inPot2, T* outPot) {
    outPot[0] = -(inPot2[1] - inPot1[1]);
    outPot[1] = inPot2[0] - inPot1[0];
    outPot[2] = inPot1[2];
}

}  // namespace Utils
