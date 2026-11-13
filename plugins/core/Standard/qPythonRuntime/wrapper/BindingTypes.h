// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// All binding translation units must see identical converters before instantiation.
// cccorelib.h owns opaque reference semantics; casters.h owns Qt conversions.
// Combine existing declarations without changing the slots macro for later Qt code.
// Initialize Qt first: restoring an undefined slots state before its first include
// would let Qt include guards prevent later QObject consumers from defining it.
#include <QtCore/qobjectdefs.h>
#pragma push_macro("slots")
#undef slots
#include "cccorelib/src/cccorelib.h"
#include "pycc/src/casters.h"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#pragma pop_macro("slots")
