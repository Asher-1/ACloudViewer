// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// std
#include <stdexcept>

class MplaneInvalidArgument : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

class MplaneFittingError : public std::logic_error {
    using std::logic_error::logic_error;
};
