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
//#                   COPYRIGHT: Daniel Girardeau-Montaut                  #
//#                                                                        #
//##########################################################################

#ifndef ECV_GEOM_FEATURES_DIALOG_HEADER
#define ECV_GEOM_FEATURES_DIALOG_HEADER

//Local
#include "ecvLibAlgorithms.h"

//Qt
#include <QDialog>

#include <ui_geomFeaturesDlg.h>

//! Dialog for computing the density of a point clouds
class ccGeomFeaturesDlg: public QDialog, public Ui::GeomFeaturesDialog
{
public:

	//! Default constructor
	explicit ccGeomFeaturesDlg(QWidget* parent = nullptr);

	//! Sets selected features
	void setSelectedFeatures(const ccLibAlgorithms::GeomCharacteristicSet& features);
	//! Returns selected features
	bool getSelectedFeatures(ccLibAlgorithms::GeomCharacteristicSet& features) const;
	//! Sets the default kernel radius (for 'precise' mode only)
	void setRadius(double r);
	//! Returns	the kernel radius (for 'precise' mode only)
	double getRadius() const;

	//! reset the whole dialog
	void reset();

protected:

	struct Option : ccLibAlgorithms::GeomCharacteristic
	{
		Option(QCheckBox* cb, cloudViewer::GeometricalAnalysisTools::GeomCharacteristic c, int option = 0)
			: ccLibAlgorithms::GeomCharacteristic(c, option)
			, checkBox(cb)
		{}

		QCheckBox* checkBox = nullptr;
	};
	
	std::vector<Option> m_options;
};

#endif // ECV_GEOM_FEATURES_DIALOG_HEADER
