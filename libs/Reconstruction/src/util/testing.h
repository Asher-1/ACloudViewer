// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// The Reconstruction (COLMAP fork) test suite uses googletest, matching the
// upstream COLMAP test files and the rest of the repository. Legacy files
// that only define TEST_NAME keep compiling: the macro is unused by gtest
// but documents the suite origin.

#pragma once

#ifndef TEST_NAME
#error "TEST_NAME not defined"
#endif

#include <gtest/gtest.h>
