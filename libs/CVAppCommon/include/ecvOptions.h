// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVAppCommon.h"

// CV_CORE_LIB
#include <CVLog.h>

// Qt
#include <QString>

/**
 * @class ecvOptions
 * @brief Main application options manager (singleton)
 * 
 * Manages global application options and preferences including:
 * - Display settings (normals visibility, etc.)
 * - Dialog behavior (native vs custom dialogs)
 * - Logging verbosity levels
 * - User interaction preferences
 * 
 * Options are automatically persisted to disk and restored on
 * application restart. Uses singleton pattern for global access.
 * 
 * @see ecvSettingManager
 */
class CVAPPCOMMON_LIB_API ecvOptions {
public:  // parameters
    /// Whether to display normals by default for loaded point clouds
    bool normalsDisplayedByDefault;

    /// Use native OS file dialogs instead of Qt dialogs
    bool useNativeDialogs;

    /// Console/log verbosity level (see CVLog::MessageLevelFlags)
    CVLog::MessageLevelFlags logVerbosityLevel;

    /// Show confirmation dialog before quitting application
    bool askForConfirmationBeforeQuitting;

public:  // methods
    /**
     * @brief Default constructor
     * 
     * Initializes options with default values.
     */
    ecvOptions();

    /**
     * @brief Reset all parameters to default values
     */
    void reset();

    /**
     * @brief Load options from persistent storage
     * 
     * Reads saved options from application settings file.
     */
    void fromPersistentSettings();

    /**
     * @brief Save options to persistent storage
     * 
     * Writes current options to application settings file.
     */
    void toPersistentSettings() const;

public:  // static methods
    /**
     * @brief Get singleton instance (const version)
     * @return Const reference to global options instance
     */
    static const ecvOptions& Instance() { return InstanceNonConst(); }

    /**
     * @brief Release singleton instance
     * 
     * Clears the singleton instance. Typically called at shutdown.
     */
    static void ReleaseInstance();

    /**
     * @brief Set options from another instance
     * 
     * Copies settings from provided options object to singleton.
     * @param options Options to copy
     */
    static void Set(const ecvOptions& options);

protected:  // methods
    /**
     * @brief Get singleton instance (non-const version)
     * @return Reference to global options instance
     */
    static ecvOptions& InstanceNonConst();
};
