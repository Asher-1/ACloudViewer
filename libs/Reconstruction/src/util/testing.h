// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UTIL_TESTING_H_
#define COLMAP_SRC_UTIL_TESTING_H_

#include <iostream>

#define BOOST_TEST_MAIN

#ifndef TEST_NAME
#error "TEST_NAME not defined"
#endif

#define BOOST_TEST_MODULE TEST_NAME

#include <boost/test/unit_test.hpp>

#endif  // COLMAP_SRC_UTIL_TESTING_H_
