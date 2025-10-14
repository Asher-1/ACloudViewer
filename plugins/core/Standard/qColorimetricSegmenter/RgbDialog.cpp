// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RgbDialog.h"

// common
#include <ecvGenericPointCloud.h>
#include <ecvPickingHub.h>

// Qt
#include <QCheckBox>

/*
        Constructor
*/
RgbDialog::RgbDialog(ccPickingHub* pickingHub, QWidget* parent)
    : QDialog(parent),
      Ui::RgbDialog(),
      m_pickingWin(nullptr),
      m_pickingHub(pickingHub) {
    assert(pickingHub);

    setModal(false);
    setupUi(this);

    // Link between Ui and actions
    connect(pointPickingButton_first, &QCheckBox::toggled, this,
            &RgbDialog::pickPoint_first);
    connect(pointPickingButton_second, &QCheckBox::toggled, this,
            &RgbDialog::pickPoint_second);

    // auto disable picking mode on quit
    connect(this, &QDialog::finished, [&]() {
        if (pointPickingButton_first->isChecked())
            pointPickingButton_first->setChecked(false);
        if (pointPickingButton_second->isChecked())
            pointPickingButton_second->setChecked(false);
    });
}

/*
        Method for the first picking point functionnality
*/
void RgbDialog::pickPoint_first(bool state) {
    if (!m_pickingHub) {
        return;
    }
    if (state) {
        if (!m_pickingHub->addListener(this, true)) {
            CVLog::Error(
                    "Can't start the picking process (another tool is using "
                    "it)");
            state = false;
        }
    } else {
        m_pickingHub->removeListener(this);
    }
    pointPickingButton_first->blockSignals(true);
    pointPickingButton_first->setChecked(state);
    pointPickingButton_first->blockSignals(false);
}

/*
        Method for the second picking point functionnality
*/
void RgbDialog::pickPoint_second(bool state) {
    if (!m_pickingHub) {
        return;
    }
    if (state) {
        if (!m_pickingHub->addListener(this, true)) {
            CVLog::Error(
                    "Can't start the picking process (another tool is using "
                    "it)");
            state = false;
        }
    } else {
        m_pickingHub->removeListener(this);
    }
    pointPickingButton_second->blockSignals(true);
    pointPickingButton_second->setChecked(state);
    pointPickingButton_second->blockSignals(false);
}

/*
        Method applied after a point is picked by picking point functionnality
*/
void RgbDialog::onItemPicked(const PickedItem& pi) {
    assert(pi.entity);
    m_pickingWin = m_pickingHub->activeWindow();

    if (pi.entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        if (static_cast<ccGenericPointCloud*>(pi.entity)->hasColors()) {
            // Get RGB values of the picked point
            ccGenericPointCloud* cloud =
                    static_cast<ccGenericPointCloud*>(pi.entity);
            const ecvColor::Rgb& rgb = cloud->getPointColor(pi.itemIndex);
            if (pointPickingButton_first->isChecked()) {
                CVLog::Print("Point picked from first point picker");

                red_first->setValue(rgb.r);
                green_first->setValue(rgb.g);
                blue_first->setValue(rgb.b);

                pointPickingButton_first->setChecked(false);
            } else {
                CVLog::Print("Point picked from second point picker");
                red_second->setValue(rgb.r);
                green_second->setValue(rgb.g);
                blue_second->setValue(rgb.b);

                pointPickingButton_second->setChecked(false);
            }
        } else {
            CVLog::Print("The point cloud is not with RGB values.");
        }
    }
}
