//##########################################################################
//#                                                                        #
//#                    CLOUDVIEWER  PLUGIN: qMPlane                        #
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

#ifndef CC_MPLANE_POINT
#define CC_MPLANE_POINT

// ECV_DB_LIB
#include "ecv2DLabel.h"


class ccMPlanePoint {

public:
	explicit ccMPlanePoint(cc2DLabel *label) : m_label(label), m_distance(0.0) {}

	unsigned int getIndex() const;
	const CCVector3& getCoordinates() const;
	cc2DLabel* getLabel();
	QString getName() const;
	void setName(const QString& newName);
	float getDistance() const;
	void setDistance(float);

private:
	cc2DLabel *m_label;
	float m_distance;
};

#endif