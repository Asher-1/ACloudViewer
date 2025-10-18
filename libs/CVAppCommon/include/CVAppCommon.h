// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore/QtGlobal>

#if defined(CVAPPCOMMON_LIBRARY_BUILD)
#define CVAPPCOMMON_LIB_API Q_DECL_EXPORT
#else
#define CVAPPCOMMON_LIB_API Q_DECL_IMPORT
#endif
