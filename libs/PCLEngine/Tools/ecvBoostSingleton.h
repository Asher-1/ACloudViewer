// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
template <class T>
struct ecvBoostSingleton {
    //! Default constructor
    ecvBoostSingleton() : instance(nullptr) {}
    //! Current instance
    std::shared_ptr<T> instance;
    //! Destructor
    ~ecvBoostSingleton() = default;
    //! Releases the current instance
};
