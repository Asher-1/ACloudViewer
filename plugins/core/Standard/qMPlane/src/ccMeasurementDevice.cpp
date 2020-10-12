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

// CC
#include "DistanceComputationTools.h"

// Local dependencies
#include "ccMeasurementDevice.h"
#include "ccMPlaneErrors.h"

constexpr char * MPLANE_SCALARFIELD_NAME = "MPlane Distance";


ccPlane* ccMeasurementDevice::fitPlaneToPoints(const std::vector<ccMPlanePoint>& fittingPoints, const QString& planeName) const
{
	ccPointCloud planeCloud;
	planeCloud.reserveThePointsTable(static_cast<unsigned int>(fittingPoints.size()));
	std::for_each(
		fittingPoints.cbegin(),
		fittingPoints.cend(),
		[&planeCloud](ccMPlanePoint point) mutable {
			planeCloud.addPoint(point.getCoordinates());
		}
	);

	ccPlane *plane = ccPlane::Fit(&planeCloud);

	if (plane) {
		setupPlaneUiDisplay(plane, planeName);
		return plane;
	}
	else {
		throw MplaneFittingError("Could not fit plane");
	}
}

void ccMeasurementDevice::createScalarFieldForCloud(const ccPlane *plane, ccPointCloud *cloud, bool signedMeasurement) const
{
	CCVector3 N = plane->getNormal();
	PointCoordinateType equation[4] = { N.x, N.y, N.z, 0 };
	plane->getEquation(N, equation[3]);

	ccScalarField *scalarFieldPlane;
	int scalarIndex;
	std::tie(scalarFieldPlane, scalarIndex) = findOrCreateScalarfieldForCloud(cloud);
	scalarFieldPlane->clear();

	addPointDistancesToScalarfield(scalarFieldPlane, cloud, equation, signedMeasurement);

	scalarFieldPlane->computeMinAndMax();
	cloud->setCurrentDisplayedScalarField(scalarIndex);
	cloud->showSFColorsScale(true);
	cloud->showSF(true);
	//cloud->setDisplay(m_app->getActiveGLWindow());
	//cloud->prepareDisplayForRefresh();
	//cloud->refreshDisplay();
	cloud->redrawDisplay();
}

void ccMeasurementDevice::deleteScalarFieldFromCloud(ccPointCloud * cloud) const
{
	int scalarIndex = cloud->getScalarFieldIndexByName(MPLANE_SCALARFIELD_NAME);
	if (scalarIndex != -1) {
		cloud->deleteScalarField(scalarIndex);
		//cloud->setDisplay(m_app->getActiveGLWindow());
		//cloud->prepareDisplayForRefresh();
		//cloud->refreshDisplay();
		cloud->redrawDisplay();
	}
}

float ccMeasurementDevice::measurePointToPlaneDistance(const ccPlane * plane, const CCVector3 & point, bool signedMeasurement) const
{
	CCVector3 N = plane->getNormal();
	PointCoordinateType equation[4] = { N.x, N.y, N.z, 0 };
	plane->getEquation(N, equation[3]);
	if (signedMeasurement) {
		return CVLib::DistanceComputationTools::computePoint2PlaneDistance(&point, equation);
	}
	else {
		return abs(CVLib::DistanceComputationTools::computePoint2PlaneDistance(&point, equation));
	}
}

void ccMeasurementDevice::setupPlaneUiDisplay(ccPlane * plane, const QString planeName) const
{
	plane->setColor(ecvColor::Rgb(255, 255, 255));
	plane->setName(planeName);
	plane->enableStippling(true);
	plane->showColors(true);
	plane->setOpacity(0.5);
	plane->applyGLTransformation_recursive();
	plane->setVisible(true);
	plane->setSelectionBehavior(ccHObject::SELECTION_IGNORED);
}

std::tuple<ccScalarField*, int> ccMeasurementDevice::findOrCreateScalarfieldForCloud(ccPointCloud *cloud) const
{
	ccScalarField *scalarFieldPlane = nullptr;

	int scalarIndex = cloud->getScalarFieldIndexByName(MPLANE_SCALARFIELD_NAME);
	if (scalarIndex == -1) {
		scalarFieldPlane = new ccScalarField(MPLANE_SCALARFIELD_NAME);
		scalarFieldPlane->reserve(cloud->size());
		scalarIndex = cloud->addScalarField(scalarFieldPlane);
	}
	else {
		scalarFieldPlane = static_cast<ccScalarField*>(cloud->getScalarField(scalarIndex));
	}
	return std::make_tuple(scalarFieldPlane, scalarIndex);
}

void ccMeasurementDevice::addPointDistancesToScalarfield(ccScalarField *scalarField, const ccPointCloud *cloud, const PointCoordinateType * equation, bool signedMeasurement) const
{
	for (unsigned int i = 0; i < cloud->size(); ++i) {
		const CCVector3 *P = cloud->getPoint(i);
		if (signedMeasurement) {
			scalarField->addElement(CVLib::DistanceComputationTools::computePoint2PlaneDistance(P, equation));
		}
		else {
			scalarField->addElement(abs(CVLib::DistanceComputationTools::computePoint2PlaneDistance(P, equation)));
		}
	}
}
