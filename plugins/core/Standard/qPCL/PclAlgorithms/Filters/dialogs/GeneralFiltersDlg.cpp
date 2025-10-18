// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "GeneralFiltersDlg.h"

// ECV_DB_LIB
#include <ecvMainAppInterface.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

GeneralFiltersDlg::GeneralFiltersDlg(ecvMainAppInterface* app)
    : QDialog(app ? app->getActiveWindow() : 0),
      Ui::GeneralFiltersDlg(),
      m_app(app) {
    setupUi(this);
    buttonGroup->setExclusive(true);
    buttonGroup->setId(curvatureRadioButton, 0);
    buttonGroup->setId(xRadioButton, 1);
    buttonGroup->setId(yRadioButton, 2);
    buttonGroup->setId(zRadioButton, 3);

    buttonGroup_2->setExclusive(true);
    buttonGroup_2->setId(curvaturePassRadioButton, 0);
    buttonGroup_2->setId(xPassRadioButton, 1);
    buttonGroup_2->setId(yPassRadioButton, 2);
    buttonGroup_2->setId(zPassRadioButton, 3);
}

ccPolyline* GeneralFiltersDlg::getPolyline() {
    // return the cloud currently selected in the combox box
    if (selectPolylineCheckBox->isChecked()) {
        return getPolylineFromCombo(polylineComboBox, m_app->dbRootObject());
    } else {
        return nullptr;
    }
}

void GeneralFiltersDlg::getContour(std::vector<CCVector3>& contour) {
    contour.clear();
    contour.push_back(CCVector3(ltxDoubleSpinBox->value(),
                                ltyDoubleSpinBox->value(),
                                ltzDoubleSpinBox->value()));
    contour.push_back(CCVector3(rtxDoubleSpinBox->value(),
                                rtyDoubleSpinBox->value(),
                                rtzDoubleSpinBox->value()));
    contour.push_back(CCVector3(lbxDoubleSpinBox->value(),
                                lbyDoubleSpinBox->value(),
                                lbzDoubleSpinBox->value()));
    contour.push_back(CCVector3(rbxDoubleSpinBox->value(),
                                rbyDoubleSpinBox->value(),
                                rbzDoubleSpinBox->value()));
}

void GeneralFiltersDlg::refreshPolylineComboBox() {
    if (m_app) {
        // add list of labels to the combo-boxes
        ccHObject::Container labels;
        if (m_app->dbRootObject())
            m_app->dbRootObject()->filterChildren(labels, true,
                                                  CV_TYPES::POLY_LINE);

        unsigned polylineCount = 0;
        polylineComboBox->clear();
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i]->isA(CV_TYPES::POLY_LINE))  // as filterChildren only
                                                      // test 'isKindOf'
            {
                QString name = getEntityName(labels[i]);
                QVariant uniqueID(labels[i]->getUniqueID());
                polylineComboBox->addItem(name, uniqueID);
                ++polylineCount;
            }
        }

        if (polylineCount >= 1 && m_app) {
            // return the 2D Label currently selected in the combox box
            selectPolylineCheckBox->setChecked(true);
            polylineComboBox->setEnabled(true);
        } else {
            selectPolylineCheckBox->setChecked(false);
            polylineComboBox->setEnabled(false);
        }
    }
}

QString GeneralFiltersDlg::getEntityName(ccHObject* obj) {
    if (!obj) {
        assert(false);
        return QString();
    }

    QString name = obj->getName();
    if (name.isEmpty()) name = tr("unnamed");
    name += QString(" [ID %1]").arg(obj->getUniqueID());

    return name;
}

ccPolyline* GeneralFiltersDlg::getPolylineFromCombo(QComboBox* comboBox,
                                                    ccHObject* dbRoot) {
    assert(comboBox && dbRoot);
    if (!comboBox || !dbRoot) {
        return nullptr;
    }

    // return the cloud currently selected in the combox box
    int index = comboBox->currentIndex();
    if (index < 0) {
        return nullptr;
    }
    unsigned uniqueID = comboBox->itemData(index).toUInt();
    ccHObject* item = dbRoot->find(uniqueID);
    if (!item || !item->isA(CV_TYPES::POLY_LINE)) {
        return nullptr;
    }
    return static_cast<ccPolyline*>(item);
}

const QString GeneralFiltersDlg::getComparisonField(float& minValue,
                                                    float& maxValue) {
    int index = tab->currentIndex();
    if (index == 0) {
        minValue = static_cast<float>(minLimitSpinBox->value());
        maxValue = static_cast<float>(maxLimitSpinBox->value());
        return buttonGroup_2->checkedButton()->text();
    } else if (index == 1) {
        minValue = static_cast<float>(minMagnitudeSpinBox->value());
        maxValue = static_cast<float>(maxMagnitudeSpinBox->value());
        return buttonGroup->checkedButton()->text();
    } else {
        assert(false);
        return "";
    }
}

void GeneralFiltersDlg::getComparisonTypes(QStringList& types) {
    types.clear();
    if (equalCheckBox->isChecked()) {
        if (greaterCheckBox->isChecked()) {
            types << "GE";
        }

        if (lessThanCheckBox->isChecked()) {
            types << "LE";
        }

        if (!greaterCheckBox->isChecked() && !lessThanCheckBox->isChecked()) {
            types << "EQ";
        }
    } else {
        if (greaterCheckBox->isChecked()) {
            types << "GT";
        }

        if (lessThanCheckBox->isChecked()) {
            types << "LT";
        }
    }
}
