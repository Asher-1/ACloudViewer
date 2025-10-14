// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_SINGLETON_HEADER
#define ECV_SINGLETON_HEADER

//! Generic singleton encapsulation structure
template <class T>
struct ecvSingleton {
    //! Default constructor
    ecvSingleton() : instance(nullptr) {}
    //! Destructor
    ~ecvSingleton() { release(); }
    //! Releases the current instance
    inline void release() {
        if (instance) {
            delete instance;
            instance = nullptr;
        }
    }

    //! Current instance
    T* instance;
};
#endif  // ECV_SINGLETON_HEADER
