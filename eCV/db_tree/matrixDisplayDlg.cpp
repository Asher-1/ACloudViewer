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

#include "matrixDisplayDlg.h"

//local
#include "ecvPersistentSettings.h"

// ECV_DB_LIB
#include <ecvFileUtils.h>
#include <ecvGuiParameters.h>

// Qt
#include <QFileDialog>
#include <QSettings>
#include <QClipboard>

MatrixDisplayDlg::MatrixDisplayDlg(QWidget* parent/*=0*/)
	: QWidget(parent)
	, Ui::MatrixDisplayDlg()
{
	setupUi(this);

	connect(exportToAsciiPushButton, &QAbstractButton::clicked, this, &MatrixDisplayDlg::exportToASCII);
	connect(exportToClipboardPushButton, &QAbstractButton::clicked, this, &MatrixDisplayDlg::exportToClipboard);

	show();
}

void MatrixDisplayDlg::fillDialogWith(const ccGLMatrix& mat)
{
	m_mat = ccGLMatrixd(mat.data());

	int precision = ecvGui::Parameters().displayedNumPrecision;

	//display as 4x4 matrix
	maxTextEdit->setText(mat.toString(precision));

	//display as rotation vector/angle
	{
		PointCoordinateType angle_rad;
		CCVector3 axis3D, t3D;
		mat.getParameters(angle_rad, axis3D, t3D);

		fillDialogWith(CCVector3d::fromArray(axis3D.u),angle_rad,CCVector3d::fromArray(t3D.u),precision);
	}
}

void MatrixDisplayDlg::fillDialogWith(const ccGLMatrixd& mat)
{
	m_mat = mat;

	int precision = ecvGui::Parameters().displayedNumPrecision;

	//display as 4x4 matrix
	maxTextEdit->setText(mat.toString(precision));

	//display as rotation vector/angle
	{
		double angle_rad;
		CCVector3d axis3D, t3D;
		mat.getParameters(angle_rad, axis3D, t3D);

		fillDialogWith(axis3D,angle_rad,t3D,precision);
	}
}

void MatrixDisplayDlg::fillDialogWith(const CCVector3d& axis, double angle_rad, const CCVector3d& T, int precision)
{
	//center position
	QString centerStr = QString("%0 ; %1 ; %2").arg(T.x,0,'f',precision).arg(T.y,0,'f',precision).arg(T.z,0,'f',precision);
	//rotation axis
	QString axisStr = QString("%0 ; %1 ; %2").arg(axis.x,0,'f',precision).arg(axis.y,0,'f',precision).arg(axis.z,0,'f',precision);
	//rotation angle
    QString angleStr = QString("%1 deg.").arg(cloudViewer::RadiansToDegrees(angle_rad), 0, 'f', precision);

	axisLabel->setText(axisStr);
	angleLabel->setText(angleStr);
	centerLabel->setText(centerStr);
}

void MatrixDisplayDlg::clear()
{
	maxTextEdit->setText("Invalid transformation");
	axisLabel->setText(QString());
	angleLabel->setText(QString());
	centerLabel->setText(QString());
	m_mat.toZero();
}

void MatrixDisplayDlg::exportToASCII()
{
	//persistent settings
	QSettings settings;
	settings.beginGroup(ecvPS::LoadFile()); //use the same folder as the load one
	QString currentPath = settings.value(ecvPS::CurrentPath(), ecvFileUtils::defaultDocPath()).toString();

	QString outputFilename = QFileDialog::getSaveFileName(this, "Select output file", currentPath, "*.mat.txt");
	if (outputFilename.isEmpty())
		return;

	if (m_mat.toAsciiFile(outputFilename))
	{
		CVLog::Print(QString("[I/O] Matrix saved as '%1'").arg(outputFilename));
	}
	else
	{
		CVLog::Error(QString("Failed to save matrix as '%1'").arg(outputFilename));
	}

	//save last saving location
	settings.setValue(ecvPS::CurrentPath(),QFileInfo(outputFilename).absolutePath());
	settings.endGroup();
}

void MatrixDisplayDlg::exportToClipboard()
{
	QClipboard* clipboard = QApplication::clipboard();
	if (clipboard)
	{
		clipboard->setText(m_mat.toString());
		CVLog::Print("Matrix saved to clipboard!");
	}
}
