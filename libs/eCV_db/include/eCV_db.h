// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_DB_HEADER
#define ECV_DB_HEADER

#include <QtCore/QtGlobal>

#if defined(ECV_DB_LIBRARY_BUILD)
#define ECV_DB_LIB_API Q_DECL_EXPORT
#else
#define ECV_DB_LIB_API Q_DECL_IMPORT
#endif

#endif  // ECV_DB_HEADER
