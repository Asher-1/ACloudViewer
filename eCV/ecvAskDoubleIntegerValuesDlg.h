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

#ifndef ECV_ASK_DOUBLE_INTEGER_VALUES_DIALOG_HEADER
#define ECV_ASK_DOUBLE_INTEGER_VALUES_DIALOG_HEADER

#include <ui_askDoubleIntegerValuesDlg.h>

//! Dialog to input 2 values with custom labels
class ecvAskDoubleIntegerValuesDlg : public QDialog, public Ui::AskDoubleIntegerValuesDialog
{
	Q_OBJECT
	
public:
	//! Default constructor
	ecvAskDoubleIntegerValuesDlg(
		const char* vName1,
		const char* vName2,
		double minVal,
		double maxVal,
		int minInt,
		int maxInt,
		double defaultVal1,
		int defaultVal2,
		int precision = 6,
		const char* windowTitle = 0,
		QWidget* parent = 0);
};

#endif // ECV_ASK_DOUBLE_INTEGER_VALUES_DIALOG_HEADER
