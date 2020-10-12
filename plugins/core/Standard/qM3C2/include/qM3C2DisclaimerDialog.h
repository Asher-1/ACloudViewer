//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER  PLUGIN: qM3C2                       #
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
//#            COPYRIGHT: UNIVERSITE EUROPEENNE DE BRETAGNE                #
//#                                                                        #
//##########################################################################

#ifndef M3C2_DISCLAIMER_DIALOG_HEADER
#define M3C2_DISCLAIMER_DIALOG_HEADER

//Qt
#include <QDialog>

class ecvMainAppInterface;

namespace Ui {
	class DisclaimerDialog;
}

//! Dialog for displaying the M3C2/UEB disclaimer
class DisclaimerDialog : public QDialog
{
public:
	//! Default constructor
	DisclaimerDialog(QWidget* parent = nullptr);
	~DisclaimerDialog();

	static bool show(ecvMainAppInterface* app);

private:
	//whether disclaimer has already been displayed (and accepted) or not	
	static bool s_disclaimerAccepted;

	Ui::DisclaimerDialog* m_ui;
};

#endif // M3C2_DISCLAIMER_DIALOG_HEADER
