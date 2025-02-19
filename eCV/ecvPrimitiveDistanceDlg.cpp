//##########################################################################
//#                                                                        #
//#                              ACloudViewer                           #
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
//#          COPYRIGHT: Chris Brown                                        #
//#                                                                        #
//##########################################################################

#include "ecvPrimitiveDistanceDlg.h"

//Qt
#include <QHeaderView>
#include <QMessageBox>


//System
#include <assert.h>

static bool s_signedDist = true;
static bool s_flipNormals = false;
static bool s_treatAsBounded = false;
ecvPrimitiveDistanceDlg::ecvPrimitiveDistanceDlg(QWidget* parent)
	: QDialog(parent, Qt::Tool)
	, Ui::primitiveDistanceDlg()
{
	setupUi(this);

	signedDistCheckBox->setChecked(s_signedDist);
	flipNormalsCheckBox->setEnabled(s_signedDist);
	flipNormalsCheckBox->setChecked(s_flipNormals);
	treatPlanesAsBoundedCheckBox->setUpdatesEnabled(false);
	treatPlanesAsBoundedCheckBox->setChecked(s_treatAsBounded);
	connect(cancelButton, &QPushButton::clicked, this, &ecvPrimitiveDistanceDlg::cancelAndExit);
	connect(okButton, &QPushButton::clicked, this, &ecvPrimitiveDistanceDlg::applyAndExit);
	connect(signedDistCheckBox, &QCheckBox::toggled, this, &ecvPrimitiveDistanceDlg::toggleSigned);
}

void ecvPrimitiveDistanceDlg::applyAndExit()
{
	s_signedDist = signedDistances();
	s_flipNormals = flipNormals();
	s_treatAsBounded = treatPlanesAsBounded();
	accept();
}

void ecvPrimitiveDistanceDlg::cancelAndExit()
{
	reject();
}

void ecvPrimitiveDistanceDlg::toggleSigned(bool state)
{
	flipNormalsCheckBox->setEnabled(state);
}

