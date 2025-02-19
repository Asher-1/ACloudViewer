//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

// LOCAL
#include "ecvPrimitiveFactoryDlg.h"
#include <MainWindow.h>

//Qt
#include <QClipboard>

// ECV_DB_LIB
#include <CVConst.h>
#include <ecvGenericPrimitive.h>
#include <ecvDisplayTools.h>
#include <ecvPlane.h>
#include <ecvBox.h>
#include <ecvSphere.h>
#include <ecvCylinder.h>
#include <ecvCone.h>
#include <ecvTorus.h>
#include <ecvDish.h>
#include <ecvCoordinateSystem.h>

//system
#include <assert.h>

ecvPrimitiveFactoryDlg::ecvPrimitiveFactoryDlg(MainWindow* win)
	: QDialog(win)
	, Ui::PrimitiveFactoryDlg()
	, m_win(win)
{
	assert(m_win);

	setupUi(this);

	connect(createPushButton, &QAbstractButton::clicked, this, &ecvPrimitiveFactoryDlg::createPrimitive);
	connect(closePushButton, &QAbstractButton::clicked, this, &QDialog::accept);
	connect(spherePosFromClipboardButton, &QPushButton::clicked, this, &ecvPrimitiveFactoryDlg::setSpherePositionFromClipboard);
	connect(spherePosToOriginButton, &QPushButton::clicked, this, &ecvPrimitiveFactoryDlg::setSpherePositionToOrigin);
    connect(csSetMatrixBasedOnSelectedObjectButton, &QPushButton::clicked, this, &ecvPrimitiveFactoryDlg::setCoordinateSystemBasedOnSelectedObject);
    connect(csMatrixTextEdit, &QPlainTextEdit::textChanged, this, &ecvPrimitiveFactoryDlg::onMatrixTextChange);
    connect(csClearMatrixButton, &QPushButton::clicked, this, &ecvPrimitiveFactoryDlg::setCSMatrixToIdentity);
    setCSMatrixToIdentity();
}

void ecvPrimitiveFactoryDlg::createPrimitive()
{
	if (!m_win)
		return;

    ccGenericPrimitive* primitive = nullptr;
	switch(tabWidget->currentIndex())
	{
		//Plane
		case 0:
			{
				primitive = new ccPlane(static_cast<PointCoordinateType>(planeWidthDoubleSpinBox->value()),
										static_cast<PointCoordinateType>(planeHeightDoubleSpinBox->value()));
			}
			break;
		//Box
		case 1:
			{
				CCVector3 dims(	static_cast<PointCoordinateType>(boxDxDoubleSpinBox->value()),
								static_cast<PointCoordinateType>(boxDyDoubleSpinBox->value()),
								static_cast<PointCoordinateType>(boxDzDoubleSpinBox->value()));
				primitive = new ccBox(dims);
			}
			break;
		//Sphere
		case 2:
			{
				ccGLMatrix transMat;
				transMat.setTranslation(CCVector3f(spherePosXDoubleSpinBox->value(), spherePosYDoubleSpinBox->value(), spherePosZDoubleSpinBox->value()));
				primitive = new ccSphere(static_cast<PointCoordinateType>(sphereRadiusDoubleSpinBox->value()), &transMat);
			}
			break;
		//Cylinder
		case 3:
			{
				primitive = new ccCylinder( static_cast<PointCoordinateType>(cylRadiusDoubleSpinBox->value()),
											static_cast<PointCoordinateType>(cylHeightDoubleSpinBox->value()));
			}
			break;
		//Cone
		case 4:
			{
				primitive = new ccCone( static_cast<PointCoordinateType>(coneBottomRadiusDoubleSpinBox->value()),
										static_cast<PointCoordinateType>(coneTopRadiusDoubleSpinBox->value()),
										static_cast<PointCoordinateType>(coneHeightDoubleSpinBox->value()),
										static_cast<PointCoordinateType>(snoutGroupBox->isChecked() ? coneXOffsetDoubleSpinBox->value() : 0),
										static_cast<PointCoordinateType>(snoutGroupBox->isChecked() ? coneYOffsetDoubleSpinBox->value() : 0));
			}
			break;
		//Torus
		case 5:
			{
				primitive = new ccTorus(static_cast<PointCoordinateType>(torusInsideRadiusDoubleSpinBox->value()),
										static_cast<PointCoordinateType>(torusOutsideRadiusDoubleSpinBox->value()),
										static_cast<PointCoordinateType>(cloudViewer::DegreesToRadians(torusAngleDoubleSpinBox->value())),
										torusRectGroupBox->isChecked(),
										static_cast<PointCoordinateType>(torusRectGroupBox->isChecked() ? torusRectSectionHeightDoubleSpinBox->value() : 0));
			}
			break;
		//Dish
		case 6:
			{
				primitive = new ccDish( static_cast<PointCoordinateType>(dishRadiusDoubleSpinBox->value()),
										static_cast<PointCoordinateType>(dishHeightDoubleSpinBox->value()),
										static_cast<PointCoordinateType>(dishEllipsoidGroupBox->isChecked() ? dishRadius2DoubleSpinBox->value() : 0));
			}
			break;
        case 7:
            {
                bool valid = false;
                ccGLMatrix mat = getCSMatrix(valid);
                if (!valid)
                {
                    mat.toIdentity();
                }
                primitive = new ccCoordinateSystem(&mat);

            }
            break;
	}

	if (primitive)
	{
		m_win->addToDB(primitive, true, true, true);
        ecvDisplayTools::ResetCameraClippingRange();
	}
}

void ecvPrimitiveFactoryDlg::setSpherePositionFromClipboard()
{
	QClipboard *clipboard = QApplication::clipboard();
	if (clipboard != nullptr)
	{
		QStringList valuesStr = clipboard->text().split(QRegExp("\\s+"), QString::SkipEmptyParts);
		if (valuesStr.size() == 3)
		{
			CCVector3d vec;
			bool success;
			for (unsigned i = 0; i < 3; ++i)
			{
				vec[i] = valuesStr[i].toDouble(&success);
				if (!success)
					break;
			}
			if (success)
			{
				spherePosXDoubleSpinBox->setValue(vec.x);
				spherePosYDoubleSpinBox->setValue(vec.y);
				spherePosZDoubleSpinBox->setValue(vec.z);
			}
		}
	}
}

void ecvPrimitiveFactoryDlg::setSpherePositionToOrigin()
{
	spherePosXDoubleSpinBox->setValue(0);
	spherePosYDoubleSpinBox->setValue(0);
    spherePosZDoubleSpinBox->setValue(0);
}

void ecvPrimitiveFactoryDlg::setCoordinateSystemBasedOnSelectedObject()
{
    ccHObject::Container selectedEnt = m_win->getSelectedEntities();
    for (auto entity : selectedEnt)
    {
        csMatrixTextEdit->setPlainText(entity->getGLTransformationHistory().toString());
    }
}


void ecvPrimitiveFactoryDlg::onMatrixTextChange()
{
    bool valid = false;
    getCSMatrix(valid);
    if (valid)
    {
        CVLog::Print("Valid ccGLMatrix");
    }
}

void ecvPrimitiveFactoryDlg::setCSMatrixToIdentity()
{
    csMatrixTextEdit->blockSignals(true);
    csMatrixTextEdit->setPlainText("1.00000000 0.00000000 0.00000000 0.00000000\n0.00000000 1.00000000 0.00000000 0.00000000\n0.00000000 0.00000000 1.00000000 0.00000000\n0.00000000 0.00000000 0.00000000 1.00000000");
    csMatrixTextEdit->blockSignals(false);
}

ccGLMatrix ecvPrimitiveFactoryDlg::getCSMatrix(bool& valid)
{
    QString text = csMatrixTextEdit->toPlainText();
    if (text.contains("["))
    {
        //automatically remove anything between square brackets
        static const QRegExp squareBracketsFilter("\\[([^]]+)\\]");
        text.replace(squareBracketsFilter, "");
        csMatrixTextEdit->blockSignals(true);
        csMatrixTextEdit->setPlainText(text);
        csMatrixTextEdit->blockSignals(false);
    }
    ccGLMatrix mat = ccGLMatrix::FromString(text, valid);
    return mat;
}
