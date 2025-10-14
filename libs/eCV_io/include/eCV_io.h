// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_IO_HEADER
#define ECV_IO_HEADER

#include <QtCore/QtGlobal>

#if defined(ECV_IO_LIBRARY_BUILD)
#define ECV_IO_LIB_API Q_DECL_EXPORT
#else
#define ECV_IO_LIB_API Q_DECL_IMPORT
#endif
#endif  // ECV_IO_HEADER