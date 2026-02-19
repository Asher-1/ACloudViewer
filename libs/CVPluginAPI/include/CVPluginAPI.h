// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file CVPluginAPI.h
 * @brief CVPluginAPI library export/import definitions
 * 
 * Defines macros for proper symbol export/import when building
 * and using the CVPluginAPI shared library. This library provides
 * the plugin interface definitions for CloudViewer plugins.
 */

#pragma once

#include <QtCore/QtGlobal>

#if defined(CVPLUGIN_API_LIBRARY_BUILD)
/// Export symbols when building the CVPluginAPI library
#define CVPLUGIN_LIB_API Q_DECL_EXPORT
#else
/// Import symbols when using the CVPluginAPI library
#define CVPLUGIN_LIB_API Q_DECL_IMPORT
#endif
