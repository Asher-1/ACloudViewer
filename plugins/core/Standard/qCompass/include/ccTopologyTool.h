// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <DistanceComputationTools.h>
#include <ecvColorTypes.h>

#include "ccGeoObject.h"
#include "ccTool.h"
#include "ccTopologyRelation.h"

/*
Tool used to assign topology (timing) relationships between different
GeoObjects.
*/
class ccTopologyTool : public ccTool {
public:
    ccTopologyTool();
    virtual ~ccTopologyTool();

    // called when the tool is set to active (for initialization)
    virtual void toolActivated() override;

    // called when the tool is set to disactive (for cleanup)
    virtual void toolDisactivated() override;

    // called when the selection is changed while this tool is active
    virtual void onNewSelection(
            const ccHObject::Container& selectedEntities) override;

    // called when "Return" or "Space" is pressed, or the "Accept Button" is
    // clicked
    void accept() override;  // do nothing

    // called when the "Escape" is pressed, or the "Cancel" button is clicked
    void cancel() override;  // do nothing
protected:
    int m_firstPick = -1;  // first object of a (pairwise) topology relationship
public:
    static int RELATIONSHIP;  // used to define the topology relationship being
                              // assigned (possible values are in
                              // ccTopologyRelation)
};
