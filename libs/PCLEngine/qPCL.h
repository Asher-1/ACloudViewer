// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file qPCL.h
 * @brief PCLEngine library export/import definitions
 *
 * Defines macros for proper symbol export/import when building
 * and using the PCLEngine shared library. This library provides
 * PCL (Point Cloud Library) integration for CloudViewer.
 *
 * The PCLEngine module bridges CloudViewer and PCL, providing:
 * - PCL visualization tools
 * - Point cloud conversion utilities (CloudViewer ↔ PCL)
 * - PCL algorithm wrappers
 * - Measurement and filtering tools
 */

#include <QtCore/QtGlobal>

#if defined(ECV_PCL_ENGINE_LIBRARY_BUILD)
#define QPCL_ENGINE_LIB_API Q_DECL_EXPORT
#else
#define QPCL_ENGINE_LIB_API Q_DECL_IMPORT
#endif
