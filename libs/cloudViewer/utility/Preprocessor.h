// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// CLOUDVIEWER_FIX_MSVC_(...)
///
/// Internal helper function which defers the evaluation of the enclosed
/// expression.
///
/// Use this macro only to workaround non-compliant behaviour of the MSVC
/// preprocessor.
///
/// Note: Could be dropped in the future if the compile flag /Zc:preprocessor
/// can be applied.
#define CLOUDVIEWER_FIX_MSVC_(...) __VA_ARGS__

/// CLOUDVIEWER_CONCAT(s1, s2)
///
/// Concatenates the expanded expressions s1 and s2.
#define CLOUDVIEWER_CONCAT_IMPL_(s1, s2) s1##s2
#define CLOUDVIEWER_CONCAT(s1, s2) CLOUDVIEWER_CONCAT_IMPL_(s1, s2)

/// CLOUDVIEWER_STRINGIFY(s)
///
/// Converts the expanded expression s to a string.
#define CLOUDVIEWER_STRINGIFY_IMPL_(s) #s
#define CLOUDVIEWER_STRINGIFY(s) CLOUDVIEWER_STRINGIFY_IMPL_(s)

/// CLOUDVIEWER_NUM_ARGS(...)
///
/// Returns the number of supplied arguments.
///
/// Note: Only works for 1-10 arguments.
#define CLOUDVIEWER_GET_NTH_ARG_(...) \
    CLOUDVIEWER_FIX_MSVC_(CLOUDVIEWER_GET_NTH_ARG_IMPL_(__VA_ARGS__))
#define CLOUDVIEWER_GET_NTH_ARG_IMPL_(arg1, arg2, arg3, arg4, arg5, arg6, arg7, \
                                 arg8, arg9, arg10, N, ...)                \
    N
#define CLOUDVIEWER_REVERSE_NUM_SEQUENCE_() 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
#define CLOUDVIEWER_NUM_ARGS(...) \
    CLOUDVIEWER_GET_NTH_ARG_(__VA_ARGS__, CLOUDVIEWER_REVERSE_NUM_SEQUENCE_())

/// CLOUDVIEWER_OVERLOAD(func, ...)
///
/// Overloads the enumerated macros func1, func2, etc. based on the number of
/// additional arguments.
///
/// Example:
///
/// \code
/// #define FOO_1(x1) foo(x1)
/// #define FOO_2(x1, x2) bar(x1, x2)
/// #define FOO(...) '\'
///     CLOUDVIEWER_FIX_MSVC_(CLOUDVIEWER_OVERLOAD(FOO_, __VA_ARGS__)(__VA_ARGS__))
///
/// FOO(1)    -> foo(1)
/// FOO(2, 3) -> bar(2, 3)
/// \endcode
#define CLOUDVIEWER_OVERLOAD(func, ...) \
    CLOUDVIEWER_CONCAT(func, CLOUDVIEWER_NUM_ARGS(__VA_ARGS__))
