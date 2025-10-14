// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVAppCommon.h"

// Qt
#include <QString>

//! Main application options
class CVAPPCOMMON_LIB_API ecvOptions {
public:  // parameters
    //! Whether to display the normals by default or not
    bool normalsDisplayedByDefault;

    //! Use native load/save dialogs
    bool useNativeDialogs;

public:  // methods
    //! Default constructor
    ecvOptions();

    //! Resets parameters to default values
    void reset();

    //! Loads from persistent DB
    void fromPersistentSettings();

    //! Saves to persistent DB
    void toPersistentSettings() const;

public:  // static methods
    //! Returns the stored values of each parameter.
    static const ecvOptions& Instance() { return InstanceNonConst(); }

    //! Release unique instance (if any)
    static void ReleaseInstance();

    //! Sets parameters
    static void Set(const ecvOptions& options);

protected:  // methods
    //! Returns the stored values of each parameter.
    static ecvOptions& InstanceNonConst();
};
