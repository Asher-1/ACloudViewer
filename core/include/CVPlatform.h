// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_PLATFORM_HEADER
#define CV_PLATFORM_HEADER

// Defines the following macros (depending on the compilation platform/settings)
//	- CV_WINDOWS / CV_MAC_OS / CV_LINUX
//	- CV_ENV32 / CV_ENV64
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32)
#define CV_WINDOWS
#if defined(_WIN64)
#define CV_ENV_64
#else
#define CV_ENV_32
#endif
#else
#if defined(__APPLE__)
#define CV_MAC_OS
#else
#define CV_LINUX
#endif
#if defined(__x86_64__) || defined(__ppc64__) || defined(__arm64__)
#define CV_ENV_64
#else
#define CV_ENV_32
#endif
#endif

#endif  // CV_PLATFORM_HEADER
