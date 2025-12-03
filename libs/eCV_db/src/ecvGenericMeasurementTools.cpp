// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGenericMeasurementTools.h"

#include "ecvDisplayTools.h"

ecvGenericMeasurementTools::ecvGenericMeasurementTools(MeasurementType type)
    : m_measurementType(type), m_associatedEntity(nullptr) {}

ecvGenericMeasurementTools::~ecvGenericMeasurementTools() {
    // Empty destructor - required for vtable generation
}

void ecvGenericMeasurementTools::update() { 
    ecvDisplayTools::UpdateScreen(); 
}

