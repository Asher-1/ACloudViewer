// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvOptions.h"

#include "ecvSettingManager.h"

// eCV_db
#include <ecvSingleton.h>

//! Unique instance of ecvOptions
static ecvSingleton<ecvOptions> s_options;

ecvOptions& ecvOptions::InstanceNonConst() {
    if (!s_options.instance) {
        s_options.instance = new ecvOptions();
        s_options.instance->fromPersistentSettings();
    }

    return *s_options.instance;
}

void ecvOptions::ReleaseInstance() { s_options.release(); }

void ecvOptions::Set(const ecvOptions& params) { InstanceNonConst() = params; }

ecvOptions::ecvOptions() { reset(); }

void ecvOptions::reset() {
    normalsDisplayedByDefault = false;
    useNativeDialogs = true;
}

void ecvOptions::fromPersistentSettings() {
    normalsDisplayedByDefault =
            ecvSettingManager::getValue("Options", "normalsDisplayedByDefault",
                                        false)
                    .toBool();
    useNativeDialogs =
            ecvSettingManager::getValue("Options", "useNativeDialogs", true)
                    .toBool();
}

void ecvOptions::toPersistentSettings() const {
    ecvSettingManager::setValue("Options", "normalsDisplayedByDefault",
                                normalsDisplayedByDefault);
    ecvSettingManager::setValue("Options", "useNativeDialogs",
                                useNativeDialogs);
}
