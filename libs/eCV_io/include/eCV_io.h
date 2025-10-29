// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore/QtGlobal>

#if defined(ECV_IO_LIBRARY_BUILD)
#define ECV_IO_LIB_API Q_DECL_EXPORT
#else
#define ECV_IO_LIB_API Q_DECL_IMPORT
#endif