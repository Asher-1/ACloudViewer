// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file CV_io.h
 * @brief CV_io library export/import definitions
 *
 * Defines macros for proper symbol export/import when building
 * and using the CV_io shared library.
 */

#pragma once

#include <QtCore/QtGlobal>

#if defined(CV_IO_LIBRARY_BUILD)
/// Export symbols when building the CV_io library
#define CV_IO_LIB_API Q_DECL_EXPORT
#else
/// Import symbols when using the CV_io library
#define CV_IO_LIB_API Q_DECL_IMPORT
#endif