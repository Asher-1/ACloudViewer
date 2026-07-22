// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/* Unified symbol export macro for all AICore C API headers.
   On Windows the library uses WINDOWS_EXPORT_ALL_SYMBOLS so no __declspec is
   needed.  On GCC/Clang the visibility attribute is only applied when building
   AICore itself (AICore_BUILD); consumers get an empty macro because visibility
   on declarations is meaningless and older GCC warns about attribute placement
   after the return type. */
#if defined(_WIN32)
#define AICORE_CAPI
#elif defined(AICore_BUILD)
#define AICORE_CAPI __attribute__((visibility("default")))
#else
#define AICORE_CAPI
#endif

/* C++ classes exported from libAICore (Qt helpers, etc.). */
#if defined(_WIN32) && defined(AICore_SHARED)
#ifdef AICore_BUILD
#define AICORE_CXX_API __declspec(dllexport)
#else
#define AICORE_CXX_API __declspec(dllimport)
#endif
#elif defined(AICore_SHARED)
#define AICORE_CXX_API __attribute__((visibility("default")))
#else
#define AICORE_CXX_API
#endif
