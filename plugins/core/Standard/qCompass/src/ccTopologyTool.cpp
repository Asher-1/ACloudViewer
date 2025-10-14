// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccTopologyTool.h"

#include "ccCompass.h"

int ccTopologyTool::RELATIONSHIP = ccTopologyRelation::YOUNGER_THAN;

ccTopologyTool::ccTopologyTool() : ccTool() {}

ccTopologyTool::~ccTopologyTool() {}

// called when the selection is changed while this tool is active
void ccTopologyTool::onNewSelection(
        const ccHObject::Container& selectedEntities) {
    if (selectedEntities.size() == 0) return;

    // is selection a geoObject?
    ccGeoObject* o = ccGeoObject::getGeoObjectParent(selectedEntities[0]);
    if (o) {
        // yes it is a GeoObject - have we got a first point?
        ccHObject* first = m_app->dbRootObject()->find(m_firstPick);
        if (!first)  // no... this is the first point (m_firstPick is invalid)
        {
            m_firstPick = o->getUniqueID();

            // write instructions to screen
            ecvDisplayTools::DisplayNewMessage(
                    "Select second (younger) GeoObject.",
                    ecvDisplayTools::LOWER_LEFT_MESSAGE);
        } else  // yes.. this is the second pick
        {
            ccGeoObject* g1 = static_cast<ccGeoObject*>(
                    first);  // n.b. this *should* always be a GeoObject....

            // add topology relation!
            g1->addRelationTo(o, ccTopologyTool::RELATIONSHIP, m_app);

            // reset...
            accept();
        }
    } else {
        // no - throw error
        m_app->dispToConsole("[ccCompass] Please select a GeoObject",
                             ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
    }
}

// called when the tool is set to active (for initialization)
void ccTopologyTool::toolActivated() {
    // display instructions
    ecvDisplayTools::DisplayNewMessage("Select first (older) GeoObject.",
                                       ecvDisplayTools::LOWER_LEFT_MESSAGE);
}

// called when the tool is set to disactive (for cleanup)
void ccTopologyTool::toolDisactivated() {
    m_firstPick = -1;  // reset
}

// called when "Return" or "Space" is pressed, or the "Accept Button" is clicked
void ccTopologyTool::accept() {
    // Reset the tool
    toolDisactivated();

    // restart picking mode
    toolActivated();
}

// called when the "Escape" is pressed, or the "Cancel" button is clicked
void ccTopologyTool::cancel() { toolDisactivated(); }
