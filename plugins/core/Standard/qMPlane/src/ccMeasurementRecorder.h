//##########################################################################
//#                                                                        #
//#                    CLOUDCOMPARE PLUGIN: qMPlane                        #
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

#ifndef CC_MPLANE_MEASUREMENT
#define CC_MPLANE_MEASUREMENT

// CC
#include "ecvPointCloud.h"
#include "ecvHObject.h"
#include "ecvMainAppInterface.h"
#include "ecvPickingListener.h"
#include "ecvPlane.h"
#include "ecv2DLabel.h"

// Local dependencies
#include "ccMPlanePoint.h"


class ccMeasurementRecorder {

public:
	ccMeasurementRecorder(ccPointCloud *rootCloud, ecvMainAppInterface *app) :
		m_rootCloud(rootCloud),
		m_app(app)
	{}

	void loadDataFromSelectedCloud();

	void addFittingPoint(const ccPickingListener::PickedItem& item);
	void deleteFittingPoint(unsigned int index);
	void renameFittingPoint(const QString& newName, unsigned int fittingPointIndex);
	const std::vector<ccMPlanePoint>& getFittingPoints() const;
	unsigned int getActualFittingPointIndex() const;
	unsigned int getFittingPointAmount() const;

	void addMeasurementPoint(const ccPickingListener::PickedItem& item, float distance);
	const std::vector<ccMPlanePoint>& getMeasurementPoints() const;
	bool renameMeasurement(const QString& newName, unsigned int measurementIndex);
	void updateMeasurement(float distance, unsigned int measurementIndex);
		
	ccPlane* getPlane() const;
	void setPlane(const ccPlane *plane);
	void deletePlane();

private:
	ecvMainAppInterface *m_app = nullptr;
	ccPointCloud *m_rootCloud = nullptr;
	ccPlane *m_plane = nullptr;

	ccHObject *m_rootFolder = nullptr;
	ccHObject *m_fittingPointFolder = nullptr;
	ccHObject *m_measurementFolder = nullptr;
	
	std::vector<ccMPlanePoint> m_fittingPoints;
	std::vector<ccMPlanePoint> m_measurementPoints;

private:
	void loadFolders();
	void loadFittingPoints();
	void loadMeasurementPoints();
	void loadAndDeleteFittingPlane();
};

#endif
