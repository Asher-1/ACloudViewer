//##########################################################################
//#                                                                        #
//#                    CLOUDVIEWER PLUGIN: qMPlane                         #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#         COPYRIGHT: AIT Austrian Institute of Technology GmbH           #
//#                                                                        #
//##########################################################################

#ifndef CC_MEASUREMENT_DEVICE
#define CC_MEASUREMENT_DEVICE

// CC
#include "ecvMainAppInterface.h"
#include "ecvPointCloud.h"
#include "ecvPlane.h"
#include "ecvScalarField.h"

// Local dependecies
#include "ccMPlanePoint.h"

class ccMeasurementDevice {

public:
	explicit ccMeasurementDevice(ecvMainAppInterface *app) : m_app(app) {}

	ccPlane* fitPlaneToPoints(const std::vector<ccMPlanePoint>& fittingPoints, const QString& planeName) const;
	void createScalarFieldForCloud(const ccPlane *plane, ccPointCloud *cloud, bool signedMeasurement) const;
	void deleteScalarFieldFromCloud(ccPointCloud *cloud) const;
	float measurePointToPlaneDistance(const ccPlane *plane, const CCVector3& point, bool signedMeasurement) const;

private:
	ecvMainAppInterface *m_app;
	void setupPlaneUiDisplay(ccPlane *plane, const QString planeName) const;
	std::tuple<ccScalarField*, int> findOrCreateScalarfieldForCloud(ccPointCloud *cloud) const;
	void addPointDistancesToScalarfield(ccScalarField *scalarField, const ccPointCloud *cloud, const PointCoordinateType *equation, bool signedMeasurement) const;
};

#endif
