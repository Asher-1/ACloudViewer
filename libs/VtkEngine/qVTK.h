// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file qVTK.h
 * @brief VtkEngine library export/import definitions
 *
 * Defines macros for proper symbol export/import when building
 * and using the VtkEngine shared library. This library provides
 * VTK (Visualization Toolkit) integration for CloudViewer.
 *
 * The VtkEngine module bridges CloudViewer and VTK, providing:
 * - VTK visualization tools
 * - Point cloud conversion utilities (CloudViewer <-> VTK)
 * - Measurement and filtering tools
 * - VTK rendering extensions
 */

#include <QtCore/QtGlobal>

#if defined(ECV_VTK_ENGINE_LIBRARY_BUILD)
#define QVTK_ENGINE_LIB_API Q_DECL_EXPORT
#else
#define QVTK_ENGINE_LIB_API Q_DECL_IMPORT
#endif
