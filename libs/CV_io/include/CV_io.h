// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore/QtGlobal>

#if defined(CV_IO_LIBRARY_BUILD)
#define CV_IO_LIB_API Q_DECL_EXPORT
#else
#define CV_IO_LIB_API Q_DECL_IMPORT
#endif