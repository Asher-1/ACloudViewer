// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef QPCL_SINGLETON_HEADER
#define QPCL_SINGLETON_HEADER

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

#endif  // QPCL_SINGLETON_HEADER
