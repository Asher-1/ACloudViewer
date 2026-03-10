// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file utils.h
/// @brief Utility functions for colors, vectors, and VTK helpers.

#include <QColor>
#include <QImage>
#include <QtMath>
#include <complex>

#include "qVTK.h"

class vtkActor;
namespace Utils {

/// @param index Character index (0-25 maps to 'x'+index)
/// @return Single-character string
QString QVTK_ENGINE_LIB_API character(int index);
/// @param path File path to select in system file explorer
void QVTK_ENGINE_LIB_API explorer(const QString& path);
/// @param size Output image size
/// @return Star-shaped QImage
QImage QVTK_ENGINE_LIB_API star(const QSize& size = QSize(30, 30));

/// @param low Lower bound
/// @param high Upper bound
/// @return Random double in [low, high)
double QVTK_ENGINE_LIB_API random(int low, int high);

/// @param low Lower bound
/// @param high Upper bound
/// @return Random complex with real and imag in [low, high)
template <typename T>
inline static std::complex<T> random(int low, int high) {
    std::complex<T> c(random(low, high), random(low, high));
    return c;
}

/// @param clr QColor to convert
/// @param vtkClr Output array [r,g,b] in 0-1 range
void QVTK_ENGINE_LIB_API vtkColor(const QColor& clr, double* vtkClr);
/// @param pClr Array [r,g,b] in 0-1 range
/// @return QColor
QColor QVTK_ENGINE_LIB_API qColor(double* pClr);
/// @param clr Input color
/// @param hsv Output array [h,s,v] (h in 0-1, s and v in 0-1)
void QVTK_ENGINE_LIB_API qColor2HSV(const QColor& clr, double* hsv);

/// @class ArrayComparator
/// @brief Functor comparing two fixed-size arrays element-wise.
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

/// @class ArrayAssigner
/// @brief Functor copying rhs array to lhs.
template <typename T, int size = 3>
class ArrayAssigner {
public:
    void operator()(T* lhs, const T* rhs) {
        // memset(lhs, rhs, 3 * sizeof(T));
        for (auto i = 0; i < size; ++i) lhs[i] = rhs[i];
    }
};

/// @class ArrayInitializer
/// @brief Functor initializing array elements to a value.
template <typename T, int size = 3>
class ArrayInitializer {
public:
    void operator()(T* array, T value = T()) {
        for (auto i = 0; i < size; ++i) array[i] = value;
    }
};

/// @class Normalizer
/// @brief Functor normalizing a 3D vector to unit length.
class Normalizer {
public:
    /// @param input Input vector
    /// @param output Normalized output (unchanged if input is zero)
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

/// @param obj VTK object to delete (no-op if nullptr)
template <class T>
inline void vtkSafeDelete(T* obj) {
    if (obj) obj->Delete();
}

/// @param objList List of VTK objects to delete and clear
template <class T>
inline void vtkSafeDelete(QList<T*>& objList) {
    foreach (T* obj, objList) obj->Delete();
    objList.clear();
}

/// @return value clamped to [min, max]
template <typename T>
inline T boundedValue(const T& value, const T& min, const T& max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/// @param vector 3D vector
/// @return Euclidean magnitude
template <typename T>
static inline double module(T* vector) {
    return qSqrt(vector[0] * vector[0] + vector[1] * vector[1] +
                 vector[2] * vector[2]);
}

/// @param pot1 First 3D point
/// @param pot2 Second 3D point
/// @return Euclidean distance between points
template <typename T>
static inline double distance(T* pot1, T* pot2) {
    double dX = pot2[0] - pot1[0];
    double dY = pot2[1] - pot1[1];
    double dZ = pot2[2] - pot1[2];
    return qSqrt(dX * dX + dY * dY + dZ * dZ);
}

/// @brief Compute 2D normal vector from two points (for line perpendicular)
/// @param inPot1 First point
/// @param inPot2 Second point
/// @param outPot Output normal vector
template <typename T>
static inline void normal(T* inPot1, T* inPot2, T* outPot) {
    outPot[0] = -(inPot2[1] - inPot1[1]);
    outPot[1] = inPot2[0] - inPot1[0];
    outPot[2] = inPot1[2];
}

}  // namespace Utils
