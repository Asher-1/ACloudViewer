// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_CORE_LIB_API_H
#define CV_CORE_LIB_API_H

#include <QtCore/QtGlobal>

#if defined(CV_CORE_LIB_LIBRARY_BUILD)
#define CV_CORE_LIB_API Q_DECL_EXPORT
#else
#define CV_CORE_LIB_API Q_DECL_IMPORT
#endif

#endif  // CV_CORE_LIB_API_H
