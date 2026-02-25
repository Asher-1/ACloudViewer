// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file CVAppCommon.h
 * @brief CVAppCommon library export/import definitions
 *
 * Defines macros for proper symbol export/import when building
 * and using the CVAppCommon shared library. This library provides
 * common functionality for CloudViewer applications.
 */

#pragma once

#include <QtCore/QtGlobal>

#if defined(CVAPPCOMMON_LIBRARY_BUILD)
/// Export symbols when building the CVAppCommon library
#define CVAPPCOMMON_LIB_API Q_DECL_EXPORT
#else
/// Import symbols when using the CVAppCommon library
#define CVAPPCOMMON_LIB_API Q_DECL_IMPORT
#endif
