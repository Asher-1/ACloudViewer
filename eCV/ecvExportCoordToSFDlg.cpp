// ##########################################################################
// #                                                                        #
// #                              CLOUDVIEWER                               #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #          COPYRIGHT: EDF R&D / DAHAI LU                                 #
// #                                                                        #
// ##########################################################################

#include "ecvExportCoordToSFDlg.h"

ccExportCoordToSFDlg::ccExportCoordToSFDlg(QWidget* parent /*=0*/)
    : QDialog(parent, Qt::Tool), Ui::ExportCoordToSFDlg() {
    setupUi(this);
}

bool ccExportCoordToSFDlg::exportX() const { return xCheckBox->isChecked(); }

bool ccExportCoordToSFDlg::exportY() const { return yCheckBox->isChecked(); }

bool ccExportCoordToSFDlg::exportZ() const { return zCheckBox->isChecked(); }
