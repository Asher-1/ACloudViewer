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

#ifndef ECV_SOR_FILTER_DLG_HEADER
#define ECV_SOR_FILTER_DLG_HEADER

#include <ui_sorFilterDlg.h>

//! Dialog to choose which dimension(s) (X, Y or Z) should be exported as SF(s)
class ecvSORFilterDlg : public QDialog, public Ui::SorFilterDialog
{
	Q_OBJECT

public:

	//! Default constructor
	explicit ecvSORFilterDlg(QWidget* parent = 0);
};

#endif // ECV_SOR_FILTER_DLG_HEADER
