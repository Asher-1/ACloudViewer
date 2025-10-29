// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cassert>

// https://gcc.gnu.org/wiki/Visibility updated to use C++11 attribute syntax
// In CloudViewer, we set symbol visibility based on folder / cmake target
// through cmake. e.g. all symbols in kernel folders are hidden. These macros
// allow fine grained control over symbol visibility.
#if defined(_WIN32) || defined(__CYGWIN__)
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
