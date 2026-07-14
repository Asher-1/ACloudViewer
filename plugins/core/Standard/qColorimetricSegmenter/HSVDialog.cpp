// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "HSVDialog.h"

// Local
#include "HSV.h"

// common
#include <CVLog.h>
#include <ecvGenericPointCloud.h>
#include <ecvPickingHub.h>
#include <ecvQtHelpers.h>

// Qt
#include <QCheckBox>

// Semi-persistent parameters
static int s_lastR = 0;
static int s_lastG = 0;
static int s_lastB = 0;

HSVDialog::HSVDialog(ccPickingHub* pickingHub, QWidget* parent)
    : QDialog(parent), Ui::HSVDialog(), m_pickingHub(pickingHub) {
    assert(pickingHub);

    setupUi(this);
    setModal(false);

    red->setValue(s_lastR);
    green->setValue(s_lastG);
    blue->setValue(s_lastB);
    updateValues();  // sync hue/sat/val from restored RGB

    updateColorButton();

    connect(pointPickingButton_first, &QCheckBox::toggled, this,
            &HSVDialog::pickPoint);
    connect(red, qOverload<int>(&QSpinBox::valueChanged), this,
            &HSVDialog::updateValues);
    connect(green, qOverload<int>(&QSpinBox::valueChanged), this,
            &HSVDialog::updateValues);
    connect(blue, qOverload<int>(&QSpinBox::valueChanged), this,
            &HSVDialog::updateValues);
    connect(this, &QDialog::accepted, this, &HSVDialog::storeParameters);

    connect(this, &QDialog::finished, [&]() {
        if (m_pickingHub) m_pickingHub->removeListener(this);
    });
}

void HSVDialog::storeParameters() {
    s_lastR = red->value();
    s_lastG = green->value();
    s_lastB = blue->value();
}

void HSVDialog::updateColorButton() {
    ccQtHelpers::SetButtonColor(rgbColorToolButton,
                                QColor(red->value(), green->value(), blue->value()));
}

void HSVDialog::pickPoint(bool state) {
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

void HSVDialog::onItemPicked(const PickedItem& pi) {
    if (!pi.entity || !m_pickingHub) {
        return;
    }

    if (pointPickingButton_first->isChecked()) {
        if (pi.entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            ccGenericPointCloud* cloud =
                    static_cast<ccGenericPointCloud*>(pi.entity);
            if (cloud->hasColors()) {
                const ecvColor::Rgb& rgb = cloud->getPointColor(pi.itemIndex);

                CVLog::Print(QString("Point picked: %1 - color: R=%2 G=%3 B=%4")
                                     .arg(pi.itemIndex)
                                     .arg(rgb.r)
                                     .arg(rgb.g)
                                     .arg(rgb.b));

                red->blockSignals(true);
                green->blockSignals(true);
                blue->blockSignals(true);

                red->setValue(rgb.r);
                green->setValue(rgb.g);
                blue->setValue(rgb.b);

                red->blockSignals(false);
                green->blockSignals(false);
                blue->blockSignals(false);

                updateValues();
                pointPickingButton_first->setChecked(false);
            }
        }
    }
}

void HSVDialog::updateValues() {
    ecvColor::Rgb rgb(red->value(), green->value(), blue->value());

    Hsv hsv(rgb);
    hue->setValue(hsv.h);
    sat->setValue(hsv.s);
    val->setValue(hsv.v);

    updateColorButton();
}
