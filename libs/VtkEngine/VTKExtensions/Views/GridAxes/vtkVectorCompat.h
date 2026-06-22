// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Compatibility shims for VTK API changes across 9.x releases.
// Provides:
//   - VTK_ABI_NAMESPACE_BEGIN/END fallback  (VTK < 9.3)
//   - vtkVector arithmetic operators        (VTK < 9.4)
//
// Include this header BEFORE any use of VTK_ABI_NAMESPACE_BEGIN in
// project code, and AFTER vtkVector.h when vector arithmetic is needed.
// This file can safely be included from both .h and .cxx files.

#ifndef vtkVectorCompat_h
#define vtkVectorCompat_h

#include "vtkVector.h"
#include "vtkVersionMacros.h"

// ---- ABI namespace compatibility (absent before VTK 9.3) ----
#ifndef VTK_ABI_NAMESPACE_BEGIN
#define VTK_ABI_NAMESPACE_BEGIN
#endif
#ifndef VTK_ABI_NAMESPACE_END
#define VTK_ABI_NAMESPACE_END
#endif

// ---- vtkVector operator compatibility (absent before VTK 9.4) ----
#if VTK_VERSION_NUMBER < VTK_VERSION_CHECK(9, 4, 0)

template <typename T, int Size>
inline vtkVector<T, Size> operator+(const vtkVector<T, Size>& a,
                                    const vtkVector<T, Size>& b) {
    vtkVector<T, Size> r;
    for (int i = 0; i < Size; ++i) r[i] = a[i] + b[i];
    return r;
}

template <typename T, int Size>
inline vtkVector<T, Size> operator-(const vtkVector<T, Size>& a,
                                    const vtkVector<T, Size>& b) {
    vtkVector<T, Size> r;
    for (int i = 0; i < Size; ++i) r[i] = a[i] - b[i];
    return r;
}

template <typename T, int Size, typename S>
inline vtkVector<T, Size> operator*(const vtkVector<T, Size>& a, S s) {
    vtkVector<T, Size> r;
    for (int i = 0; i < Size; ++i) r[i] = a[i] * static_cast<T>(s);
    return r;
}

template <typename T, int Size, typename S>
inline vtkVector<T, Size> operator*(S s, const vtkVector<T, Size>& a) {
    return a * s;
}

template <typename T, int Size>
inline vtkVector<T, Size> operator/(const vtkVector<T, Size>& a,
                                    const vtkVector<T, Size>& b) {
    vtkVector<T, Size> r;
    for (int i = 0; i < Size; ++i) r[i] = a[i] / b[i];
    return r;
}

#endif  // VTK < 9.4
#endif  // vtkVectorCompat_h
