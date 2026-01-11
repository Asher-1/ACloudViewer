// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPrimitiveFactoryDlg.h"

#include <MainWindow.h>

// Qt
#include <QClipboard>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

// ECV_DB_LIB
#include <CVConst.h>
#include <ecvBox.h>
#include <ecvCone.h>
#include <ecvCoordinateSystem.h>
#include <ecvCylinder.h>
#include <ecvDisc.h>
#include <ecvDish.h>
#include <ecvDisplayTools.h>
#include <ecvGenericPrimitive.h>
#include <ecvPlane.h>
#include <ecvSphere.h>
#include <ecvTorus.h>

// system
#include <assert.h>

ecvPrimitiveFactoryDlg::ecvPrimitiveFactoryDlg(MainWindow* win)
    : QDialog(win), Ui::PrimitiveFactoryDlg(), m_win(win) {
    assert(m_win);

    setupUi(this);

    connect(createPushButton, &QAbstractButton::clicked, this,
            &ecvPrimitiveFactoryDlg::createPrimitive);
    connect(closePushButton, &QAbstractButton::clicked, this, &QDialog::accept);
    connect(spherePosFromClipboardButton, &QPushButton::clicked, this,
            &ecvPrimitiveFactoryDlg::setSpherePositionFromClipboard);
    connect(spherePosToOriginButton, &QPushButton::clicked, this,
            &ecvPrimitiveFactoryDlg::setSpherePositionToOrigin);
    connect(csSetMatrixBasedOnSelectedObjectButton, &QPushButton::clicked, this,
            &ecvPrimitiveFactoryDlg::setCoordinateSystemBasedOnSelectedObject);
    connect(csMatrixTextEdit, &QPlainTextEdit::textChanged, this,
            &ecvPrimitiveFactoryDlg::onMatrixTextChange);
    connect(csClearMatrixButton, &QPushButton::clicked, this,
            &ecvPrimitiveFactoryDlg::setCSMatrixToIdentity);
    setCSMatrixToIdentity();
}

void ecvPrimitiveFactoryDlg::createPrimitive() {
    if (!m_win) return;

    ccGenericPrimitive* primitive = nullptr;
    switch (tabWidget->currentIndex()) {
        // Plane
        case 0: {
            primitive = new ccPlane(static_cast<PointCoordinateType>(
                                            planeWidthDoubleSpinBox->value()),
                                    static_cast<PointCoordinateType>(
                                            planeHeightDoubleSpinBox->value()));
        } break;
        // Box
        case 1: {
            CCVector3 dims(static_cast<PointCoordinateType>(
                                   boxDxDoubleSpinBox->value()),
                           static_cast<PointCoordinateType>(
                                   boxDyDoubleSpinBox->value()),
                           static_cast<PointCoordinateType>(
                                   boxDzDoubleSpinBox->value()));
            primitive = new ccBox(dims);
        } break;
        // Sphere
        case 2: {
            ccGLMatrix transMat;
            transMat.setTranslation(
                    CCVector3f(spherePosXDoubleSpinBox->value(),
                               spherePosYDoubleSpinBox->value(),
                               spherePosZDoubleSpinBox->value()));
            primitive =
                    new ccSphere(static_cast<PointCoordinateType>(
                                         sphereRadiusDoubleSpinBox->value()),
                                 &transMat);
        } break;
        // Cylinder
        case 3: {
            primitive =
                    new ccCylinder(static_cast<PointCoordinateType>(
                                           cylRadiusDoubleSpinBox->value()),
                                   static_cast<PointCoordinateType>(
                                           cylHeightDoubleSpinBox->value()));
        } break;
        // Cone
        case 4: {
            primitive = new ccCone(
                    static_cast<PointCoordinateType>(
                            coneBottomRadiusDoubleSpinBox->value()),
                    static_cast<PointCoordinateType>(
                            coneTopRadiusDoubleSpinBox->value()),
                    static_cast<PointCoordinateType>(
                            coneHeightDoubleSpinBox->value()),
                    static_cast<PointCoordinateType>(
                            snoutGroupBox->isChecked()
                                    ? coneXOffsetDoubleSpinBox->value()
                                    : 0),
                    static_cast<PointCoordinateType>(
                            snoutGroupBox->isChecked()
                                    ? coneYOffsetDoubleSpinBox->value()
                                    : 0));
        } break;
        // Torus
        case 5: {
            primitive = new ccTorus(
                    static_cast<PointCoordinateType>(
                            torusInsideRadiusDoubleSpinBox->value()),
                    static_cast<PointCoordinateType>(
                            torusOutsideRadiusDoubleSpinBox->value()),
                    static_cast<PointCoordinateType>(
                            cloudViewer::DegreesToRadians(
                                    torusAngleDoubleSpinBox->value())),
                    torusRectGroupBox->isChecked(),
                    static_cast<PointCoordinateType>(
                            torusRectGroupBox->isChecked()
                                    ? torusRectSectionHeightDoubleSpinBox
                                              ->value()
                                    : 0));
        } break;
        // Dish
        case 6: {
            primitive = new ccDish(
                    static_cast<PointCoordinateType>(
                            dishRadiusDoubleSpinBox->value()),
                    static_cast<PointCoordinateType>(
                            dishHeightDoubleSpinBox->value()),
                    static_cast<PointCoordinateType>(
                            dishEllipsoidGroupBox->isChecked()
                                    ? dishRadius2DoubleSpinBox->value()
                                    : 0));
        } break;
        case 7: {
            bool valid = false;
            ccGLMatrix mat = getCSMatrix(valid);
            if (!valid) {
                mat.toIdentity();
            }
            primitive = new ccCoordinateSystem(&mat);

        } break;
        // Disc
        case 8: {
            primitive = new ccDisc(static_cast<PointCoordinateType>(
                    discRadiusDoubleSpinBox->value()));
        } break;
    }

    if (primitive) {
        m_win->addToDB(primitive, true, true, true);
        ecvDisplayTools::ResetCameraClippingRange();
    }
}

void ecvPrimitiveFactoryDlg::setSpherePositionFromClipboard() {
    QClipboard* clipboard = QApplication::clipboard();
    if (clipboard != nullptr) {
        // Use QtCompat for Qt5/Qt6 compatibility
        QStringList valuesStr = qtCompatSplitRegex(clipboard->text(), "\\s+",
                                                   QtCompat::SkipEmptyParts);
        if (valuesStr.size() == 3) {
            CCVector3d vec;
            bool success;
            for (unsigned i = 0; i < 3; ++i) {
                vec[i] = valuesStr[i].toDouble(&success);
                if (!success) break;
            }
            if (success) {
                spherePosXDoubleSpinBox->setValue(vec.x);
                spherePosYDoubleSpinBox->setValue(vec.y);
                spherePosZDoubleSpinBox->setValue(vec.z);
            }
        }
    }
}

void ecvPrimitiveFactoryDlg::setSpherePositionToOrigin() {
    spherePosXDoubleSpinBox->setValue(0);
    spherePosYDoubleSpinBox->setValue(0);
    spherePosZDoubleSpinBox->setValue(0);
}

void ecvPrimitiveFactoryDlg::setCoordinateSystemBasedOnSelectedObject() {
    ccHObject::Container selectedEnt = m_win->getSelectedEntities();
    for (auto entity : selectedEnt) {
        csMatrixTextEdit->setPlainText(
                entity->getGLTransformationHistory().toString());
    }
}

void ecvPrimitiveFactoryDlg::onMatrixTextChange() {
    bool valid = false;
    getCSMatrix(valid);
    if (valid) {
        CVLog::Print("Valid ccGLMatrix");
    }
}

void ecvPrimitiveFactoryDlg::setCSMatrixToIdentity() {
    csMatrixTextEdit->blockSignals(true);
    csMatrixTextEdit->setPlainText(
            "1.00000000 0.00000000 0.00000000 0.00000000\n0.00000000 "
            "1.00000000 0.00000000 0.00000000\n0.00000000 0.00000000 "
            "1.00000000 0.00000000\n0.00000000 0.00000000 0.00000000 "
            "1.00000000");
    csMatrixTextEdit->blockSignals(false);
}

ccGLMatrix ecvPrimitiveFactoryDlg::getCSMatrix(bool& valid) {
    QString text = csMatrixTextEdit->toPlainText();
    if (text.contains("[")) {
        // automatically remove anything between square brackets
        // Use QtCompat for Qt5/Qt6 compatibility
        // Use static const for efficiency (regex compiled only once)
        static const QtCompatRegExp squareBracketsFilter("\\[([^]]+)\\]");
        text.replace(squareBracketsFilter, "");
        csMatrixTextEdit->blockSignals(true);
        csMatrixTextEdit->setPlainText(text);
        csMatrixTextEdit->blockSignals(false);
    }
    ccGLMatrix mat = ccGLMatrix::FromString(text, valid);
    return mat;
}
