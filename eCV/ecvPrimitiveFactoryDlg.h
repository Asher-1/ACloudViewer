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

#ifndef ECV_PRIMITIVE_FACTORY_DLG_HEADER
#define ECV_PRIMITIVE_FACTORY_DLG_HEADER

#include "ui_primitiveFactoryDlg.h"

//Qt
#include <QDialog>

class MainWindow;
class ccGLMatrix;

//! Primitive factory
class ecvPrimitiveFactoryDlg : public QDialog, public Ui::PrimitiveFactoryDlg
{
	Q_OBJECT

public:

	//! Default constructor
	explicit ecvPrimitiveFactoryDlg(MainWindow* win);

protected slots:

	//! Creates currently defined primitive
	void createPrimitive();

protected:
	//! Set sphere position from clipboard
	void setSpherePositionFromClipboard();

	//! Set sphere position to origin
	void setSpherePositionToOrigin();

    void setCoordinateSystemBasedOnSelectedObject();

    void onMatrixTextChange();

    void setCSMatrixToIdentity();

    ccGLMatrix getCSMatrix(bool &valid);

protected:

	//! Associated main window
	MainWindow* m_win;

};

#endif // ECV_PRIMITIVE_FACTORY_DLG_HEADER
