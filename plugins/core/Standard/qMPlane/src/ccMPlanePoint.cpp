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


// ECV_DB_LIB
#include "ecvGenericPointCloud.h"

// Local dependencies
#include "ccMPlanePoint.h"

unsigned int ccMPlanePoint::getIndex() const {
	return m_label->getPickedPoint(0).index;
}

const CCVector3& ccMPlanePoint::getCoordinates() const {
	return *m_label->getPickedPoint(0).cloudOrVertices()->getPoint(getIndex());
}

cc2DLabel* ccMPlanePoint::getLabel() {
	return m_label;
}

QString ccMPlanePoint::getName() const {
	return m_label->getName();
}

void ccMPlanePoint::setName(const QString& newName) {
	m_label->setName(newName);
}

float ccMPlanePoint::getDistance() const 
{
	return m_distance;
}

void ccMPlanePoint::setDistance(float distance)
{
	m_distance = distance;
}
