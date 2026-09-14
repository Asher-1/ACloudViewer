// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#if defined(__has_builtin)
#if __has_builtin(__builtin_expect)
#define COLMAP_BUILTIN_EXPECT_PRESENT
#endif
#endif

#if !defined(COLMAP_BUILTIN_EXPECT_PRESENT) && defined(__GNUG__)
// __has_builtin is not available prior to GCC 10
#define COLMAP_BUILTIN_EXPECT_PRESENT
#endif

#if defined(COLMAP_BUILTIN_EXPECT_PRESENT)

#ifndef COLMAP_PREDICT_BRANCH_NOT_TAKEN
#define COLMAP_PREDICT_BRANCH_NOT_TAKEN(x) (__builtin_expect(x, 0))
#endif

#ifndef COLMAP_PREDICT_FALSE
#define COLMAP_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#endif

#ifndef COLMAP_PREDICT_TRUE
#define COLMAP_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#endif

#else

#ifndef COLMAP_PREDICT_BRANCH_NOT_TAKEN
#define COLMAP_PREDICT_BRANCH_NOT_TAKEN(x) x
#endif

#ifndef COLMAP_PREDICT_TRUE
#define COLMAP_PREDICT_FALSE(x) x
#endif

#ifndef COLMAP_PREDICT_TRUE
#define COLMAP_PREDICT_TRUE(x) x
#endif

#endif

#undef COLMAP_BUILTIN_EXPECT_PRESENT
