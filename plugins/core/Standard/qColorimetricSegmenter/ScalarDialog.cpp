// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ScalarDialog.h"

// common
#include <CVLog.h>
#include <ecvGenericPointCloud.h>
#include <ecvPickingHub.h>

// Qt
#include <QCheckBox>

ScalarDialog::ScalarDialog(ccPickingHub* pickingHub, QWidget* parent)
    : QDialog(parent), Ui::ScalarDialog(), m_pickingHub(pickingHub) {
    assert(pickingHub);

    setModal(false);
    setupUi(this);

    connect(pointPickingButton_first, &QCheckBox::toggled, this,
            &ScalarDialog::pickPoint_first);
    connect(pointPickingButton_second, &QCheckBox::toggled, this,
            &ScalarDialog::pickPoint_second);

    connect(this, &QDialog::finished, [&]() {
        if (pointPickingButton_first->isChecked())
            pointPickingButton_first->setChecked(false);
        if (pointPickingButton_second->isChecked())
            pointPickingButton_second->setChecked(false);
    });
}

void ScalarDialog::pickPoint_first(bool state) {
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

void ScalarDialog::pickPoint_second(bool state) {
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

void ScalarDialog::onItemPicked(const PickedItem& pi) {
    if (!pi.entity || !m_pickingHub) {
        return;
    }

    if (pi.entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccGenericPointCloud* cloud =
                static_cast<ccGenericPointCloud*>(pi.entity);
        if (cloud->hasDisplayedScalarField()) {
            const ScalarType scalarValue =
                    cloud->getPointScalarValue(pi.itemIndex);
            CVLog::Print(QString("%0 point picked: %1 - SF value = %2")
                                 .arg(pointPickingButton_first->isChecked()
                                              ? "First"
                                              : "Second")
                                 .arg(pi.itemIndex)
                                 .arg(scalarValue));

            if (pointPickingButton_first->isChecked()) {
                first->setValue(scalarValue);

                pointPickingButton_first->setChecked(false);
            } else {
                second->setValue(scalarValue);

                pointPickingButton_second->setChecked(false);
            }
        } else {
            CVLog::Print(
                    "This point cloud doesn't have an active scalar field");
        }
    }
}
