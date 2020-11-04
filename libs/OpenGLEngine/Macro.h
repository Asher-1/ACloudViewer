// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#define CLOUDVIEWER_CONCATENATE_IMPL(s1, s2) s1##s2
#define CLOUDVIEWER_CONCATENATE(s1, s2) CLOUDVIEWER_CONCATENATE_IMPL(s1, s2)

#if defined _WIN32 || defined __CYGWIN__
#define CLOUDVIEWER_DLL_IMPORT __declspec(dllimport)
#define CLOUDVIEWER_DLL_EXPORT __declspec(dllexport)
#else
#define CLOUDVIEWER_DLL_IMPORT
#define CLOUDVIEWER_DLL_EXPORT
#endif

#ifdef CLOUDVIEWER_ENABLE_DLL_EXPORTS
#define CLOUDVIEWER_API CLOUDVIEWER_DLL_EXPORT
#else
#define CLOUDVIEWER_API CLOUDVIEWER_DLL_IMPORT
#endif
