// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore/QtGlobal>

#if defined(ECV_PCL_ENGINE_LIBRARY_BUILD)
#define QPCL_ENGINE_LIB_API Q_DECL_EXPORT
#else
#define QPCL_ENGINE_LIB_API Q_DECL_IMPORT
#endif
