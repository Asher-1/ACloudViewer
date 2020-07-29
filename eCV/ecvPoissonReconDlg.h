//##########################################################################
//#                                                                        #
//#                  CLOUDVIEWER  PLUGIN: qPoissonRecon                    #
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
//#                  COPYRIGHT: Daniel Girardeau-Montaut                   #
//#                                                                        #
//##########################################################################

#ifndef ECV_POISSON_RECONSTRUCTION_DLG_HEADER
#define ECV_POISSON_RECONSTRUCTION_DLG_HEADER

//dialog
#include "ecvHObject.h"
#include "ui_poissonReconParametersDlg.h"

//! Wrapper to the "Poisson Surface Reconstruction (Version 9)" algorithm
/** "Poisson Surface Reconstruction", M. Kazhdan, M. Bolitho, and H. Hoppe
	Symposium on Geometry Processing (June 2006), pages 61--70
	http://www.cs.jhu.edu/~misha/Code/PoissonRecon/
**/

class ccPointCloud;
class ecvPoissonReconDlg : public QDialog, public Ui::PoissonReconParamDialog
{
	Q_OBJECT
public:
	explicit ecvPoissonReconDlg(QWidget* parent = 0);

	bool start();
	bool addEntity(ccHObject* ent);
	ccHObject::Container& getReconstructions();

protected:
	bool showDialog();
	bool doComputation();
	void updateParams();
	void adjustParams(ccPointCloud* cloud);

private:
	QWidget* m_app;
	bool m_applyAllClouds;
	ccHObject::Container m_clouds;
	ccHObject::Container m_result;
	std::vector<bool> m_normalsMask;
};

#endif // ECV_POISSON_RECONSTRUCTION_DLG_HEADER