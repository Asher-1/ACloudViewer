// Compatibility operators for vtkVector missing in VTK <= 9.3.
// ParaView's bundled VTK adds operator+ and operator/ which the
// GridAxes module relies on. Remove this file when upgrading to a
// VTK version that includes these operators natively.
#ifndef vtkVectorCompat_h
#define vtkVectorCompat_h

#include "vtkVector.h"
#include "vtkVersionMacros.h"

#if VTK_VERSION_NUMBER < VTK_VERSION_CHECK(9, 4, 0)

template <typename T, int Size>
inline vtkVector<T, Size> operator+(const vtkVector<T, Size>& a, const vtkVector<T, Size>& b)
{
  vtkVector<T, Size> r;
  for (int i = 0; i < Size; ++i)
    r[i] = a[i] + b[i];
  return r;
}

template <typename T, int Size>
inline vtkVector<T, Size> operator/(const vtkVector<T, Size>& a, const vtkVector<T, Size>& b)
{
  vtkVector<T, Size> r;
  for (int i = 0; i < Size; ++i)
    r[i] = a[i] / b[i];
  return r;
}

#endif // VTK < 9.4
#endif // vtkVectorCompat_h
