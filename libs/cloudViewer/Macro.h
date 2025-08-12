// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once
#include <cassert>

#define CLOUDVIEWER_CONCATENATE_IMPL(s1, s2) s1##s2
#define CLOUDVIEWER_CONCATENATE(s1, s2) CLOUDVIEWER_CONCATENATE_IMPL(s1, s2)

// https://gcc.gnu.org/wiki/Visibility updated to use C++11 attribute syntax
#if defined _WIN32 || defined __CYGWIN__
#define CLOUDVIEWER_DLL_IMPORT __declspec(dllimport)
#define CLOUDVIEWER_DLL_EXPORT __declspec(dllexport)
#define CLOUDVIEWER_DLL_LOCAL
#else
#define CLOUDVIEWER_DLL_IMPORT [[gnu::visibility("default")]]
#define CLOUDVIEWER_DLL_EXPORT [[gnu::visibility("default")]]
#define CLOUDVIEWER_DLL_LOCAL [[gnu::visibility("hidden")]]
#endif

#ifdef CLOUDVIEWER_STATIC
#define CLOUDVIEWER_API
#define CLOUDVIEWER_LOCAL
#else
#define CLOUDVIEWER_LOCAL CLOUDVIEWER_DLL_LOCAL
#if defined(CLOUDVIEWER_ENABLE_DLL_EXPORTS)
#define CLOUDVIEWER_API CLOUDVIEWER_DLL_EXPORT
#else
#define CLOUDVIEWER_API CLOUDVIEWER_DLL_IMPORT
#endif
#endif

// Compiler-specific function macro.
// Ref: https://stackoverflow.com/a/4384825
#ifdef _WIN32
#define CLOUDVIEWER_FUNCTION __FUNCSIG__
#else
#define CLOUDVIEWER_FUNCTION __PRETTY_FUNCTION__
#endif

// Assertion for CUDA device code.
// Usage:
//     CLOUDVIEWER_ASSERT(condition);
//     CLOUDVIEWER_ASSERT(condition && "Error message");
// For host-only code, consider using utility::LogError();
#define CLOUDVIEWER_ASSERT(...) assert((__VA_ARGS__))

// OPEN3D macro compatibility layer
#ifndef OPEN3D_STRINGIFY
#define OPEN3D_STRINGIFY(x) #x
#endif

#ifndef OPEN3D_ASSERT
#define OPEN3D_ASSERT(...) CLOUDVIEWER_ASSERT(__VA_ARGS__)
#endif

// Provide function name macro and map OPEN3D_FUNCTION
#ifndef CLOUDVIEWER_FUNCTION
#if defined(__GNUC__) || defined(__clang__)
#define CLOUDVIEWER_FUNCTION __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define CLOUDVIEWER_FUNCTION __FUNCSIG__
#else
#define CLOUDVIEWER_FUNCTION __func__
#endif
#endif

#ifndef OPEN3D_FUNCTION
#define OPEN3D_FUNCTION CLOUDVIEWER_FUNCTION
#endif

// Device / host-device qualifiers used across CloudViewer codebase
#ifndef CLOUDVIEWER_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CLOUDVIEWER_DEVICE __device__
#else
#define CLOUDVIEWER_DEVICE
#endif
#endif

#ifndef CLOUDVIEWER_HOST_DEVICE
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CLOUDVIEWER_HOST_DEVICE __host__ __device__
#else
#define CLOUDVIEWER_HOST_DEVICE
#endif
#endif

#ifndef CLOUDVIEWER_HOST
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CLOUDVIEWER_HOST __host__
#else
#define CLOUDVIEWER_HOST
#endif
#endif

// Force inline used by CUDA/SIMD helpers
#ifndef CLOUDVIEWER_FORCE_INLINE
#if defined(_MSC_VER)
#define CLOUDVIEWER_FORCE_INLINE __forceinline
#else
#define CLOUDVIEWER_FORCE_INLINE inline __attribute__((always_inline))
#endif
#endif

// Map Open3D visibility macros to CloudViewer equivalents for compatibility
#ifndef OPEN3D_API
#define OPEN3D_API CLOUDVIEWER_API
#endif
#ifndef OPEN3D_DLL_EXPORT
#define OPEN3D_DLL_EXPORT CLOUDVIEWER_DLL_EXPORT
#endif
#ifndef OPEN3D_DLL_IMPORT
#define OPEN3D_DLL_IMPORT CLOUDVIEWER_DLL_IMPORT
#endif
#ifndef OPEN3D_DLL_LOCAL
#define OPEN3D_DLL_LOCAL CLOUDVIEWER_DLL_LOCAL
#endif