// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>

#define BOOST_TEST_MAIN

#ifndef TEST_NAME
#error "TEST_NAME not defined"
#endif

#define BOOST_TEST_MODULE TEST_NAME

#include <boost/test/unit_test.hpp>
