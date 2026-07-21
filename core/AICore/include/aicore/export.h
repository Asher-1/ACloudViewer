// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/* Unified symbol export macro for all AICore C API headers. */
#if defined(_WIN32) && defined(AICore_SHARED)
#ifdef AICore_BUILD
#define AICORE_CAPI __declspec(dllexport)
#else
#define AICORE_CAPI __declspec(dllimport)
#endif
#elif defined(AICore_SHARED)
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
