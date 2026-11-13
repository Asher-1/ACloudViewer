// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/* Compiler attributes are shared by public and private API declarations. */
#if defined(_WIN32)
#define AICORE_SYMBOL_EXPORT __declspec(dllexport)
#define AICORE_SYMBOL_IMPORT __declspec(dllimport)
#else
#define AICORE_SYMBOL_EXPORT __attribute__((visibility("default")))
#define AICORE_SYMBOL_IMPORT __attribute__((visibility("default")))
#endif

/* Preserve the private upstream-style MinGW decoration contract. */
#if defined(__MINGW32__)
#define AICORE_PRIVATE_SYMBOL_EXPORT __attribute__((visibility("default")))
#define AICORE_PRIVATE_SYMBOL_IMPORT __attribute__((visibility("default")))
#else
#define AICORE_PRIVATE_SYMBOL_EXPORT AICORE_SYMBOL_EXPORT
#define AICORE_PRIVATE_SYMBOL_IMPORT AICORE_SYMBOL_IMPORT
#endif

#if defined(_MSC_VER)
#define AICORE_DEPRECATED(message) __declspec(deprecated(message))
#elif defined(__GNUC__) || defined(__clang__)
#define AICORE_DEPRECATED(message) __attribute__((deprecated(message)))
#else
#define AICORE_DEPRECATED(message)
#endif

/* Preserve printf checking without compiler branches in individual tasks. */
#if defined(__GNUC__) || defined(__clang__)
#define AICORE_PRINTF_FORMAT(format_index, first_argument) \
    __attribute__((format(printf, format_index, first_argument)))
#else
#define AICORE_PRINTF_FORMAT(format_index, first_argument)
#endif

/* Internal C++ and ggml symbols remain hidden on every platform. */
#if defined(AICore_BUILD) && (!defined(_WIN32) || defined(AICore_SHARED))
#define AICORE_CAPI AICORE_SYMBOL_EXPORT
#elif defined(_WIN32) && defined(AICore_SHARED)
#define AICORE_CAPI AICORE_SYMBOL_IMPORT
#else
#define AICORE_CAPI
#endif

/* C++ classes exported from libAICore (Qt helpers, etc.). */
#if defined(AICore_SHARED)
#ifdef AICore_BUILD
#define AICORE_CXX_API AICORE_SYMBOL_EXPORT
#else
#define AICORE_CXX_API AICORE_SYMBOL_IMPORT
#endif
#else
#define AICORE_CXX_API
#endif
