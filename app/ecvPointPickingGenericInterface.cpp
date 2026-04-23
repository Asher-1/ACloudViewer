// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPointPickingGenericInterface.h"

// Local
#include "MainWindow.h"
#include "db_tree/ecvDBRoot.h"

// CV_CORE_LIB
#include <CVLog.h>

// common
#include <ecvPickingHub.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvPointCloud.h>

ccPointPickingGenericInterface::ccPointPickingGenericInterface(
        ccPickingHub* pickingHub, QWidget* parent /*=0*/)
    : ccOverlayDialog(parent), m_pickingHub(pickingHub) {
    assert(m_pickingHub);
}

bool ccPointPickingGenericInterface::linkWith(QWidget* win) {
    if (m_associatedWin == win) {
        return true;
    }

    bool wasProcessing = m_processing;

    if (!ccOverlayDialog::linkWith(win)) {
        return false;
    }

    if (wasProcessing && win && m_pickingHub) {
        m_pickingHub->addListener(this, true, true,
                                  ecvDisplayTools::POINT_PICKING);
    }

    return true;
}

bool ccPointPickingGenericInterface::start() {
    if (!m_pickingHub) {
        CVLog::Error("[Point picking] No associated display!");
        return false;
    }

    // activate "point picking mode" in associated GL window
    if (!m_pickingHub->addListener(this, true, true,
                                   ecvDisplayTools::POINT_PICKING)) {
        CVLog::Error(
                "Picking mechanism already in use. Close the tool using it "
                "first.");
        return false;
    }

    ccOverlayDialog::start();
    return true;
}

void ccPointPickingGenericInterface::stop(bool state) {
    if (m_pickingHub) {
        // deactivate "point picking mode" in all GL windows
        m_pickingHub->removeListener(this);
    }

    ccOverlayDialog::stop(state);
}

void ccPointPickingGenericInterface::onItemPicked(const PickedItem& pi) {
    if (m_processing && pi.entity) {
        processPickedPoint(pi);
    }
}
