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

#include "PVFilter.h"

// CV_CORE_LIB
#include <CVLog.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>
#include <ecvScalarField.h>
#include <ecvProgressDialog.h>
#include <ecvHObjectCaster.h>

//Qt
#include <QFile>

PVFilter::PVFilter()
	: FileIOFilter({
					"_Point+Value Filter",
					DEFAULT_PRIORITY,	// priority
					QStringList{ "pv" },
					"pv",
					QStringList{ "Point+Value cloud (*.pv)" },
					QStringList{ "Point+Value cloud (*.pv)" },
					Import | Export
		})
{
}

bool PVFilter::canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const
{
	if (type == CV_TYPES::POINT_CLOUD)
	{
		multiple = false;
		exclusive = true;
		return true;
	}
	return false;
}

CC_FILE_ERROR PVFilter::saveToFile(ccHObject* entity, const QString& filename, const SaveParameters& parameters)
{
	if (!entity || filename.isEmpty())
		return CC_FERR_BAD_ARGUMENT;

	//the cloud to save
	ccGenericPointCloud* theCloud = ccHObjectCaster::ToGenericPointCloud(entity);
	if (!theCloud)
	{
		CVLog::Warning("[PV] This filter can only save one cloud at a time!");
		return CC_FERR_BAD_ENTITY_TYPE;
	}
	unsigned numberOfPoints = theCloud->size();

	if (numberOfPoints == 0)
	{
		CVLog::Warning("[PV] Input cloud is empty!");
		return CC_FERR_NO_SAVE;
	}

	//open binary file for writing
	QFile out(filename);
	if (!out.open(QIODevice::WriteOnly))
		return CC_FERR_WRITING;

	//Has the cloud been recentered?
	if (theCloud->isShifted())
		CVLog::Warning(QString("[PVFilter::save] Can't recenter or rescale cloud '%1' when saving it in a PN file!").arg(theCloud->getName()));

	//for point clouds with multiple SFs, we must set the currently displayed one as 'input' SF
	//if (theCloud->isA(CV_TYPES::POINT_CLOUD))
	//{
	//	ccPointCloud* pc = static_cast<ccPointCloud*>(theCloud);
	//	pc->setCurrentInScalarField(pc->getCurrentDisplayedScalarFieldIndex());
	//}
	bool hasSF = theCloud->hasDisplayedScalarField();
	if (!hasSF)
		CVLog::Warning(QString("[PVFilter::save] Cloud '%1' has no displayed scalar field (we will save points with a default scalar value)!").arg(theCloud->getName()));

	float val = std::numeric_limits<float>::quiet_NaN();

	//progress dialog
	QScopedPointer<ecvProgressDialog> pDlg(0);
	if (parameters.parentWidget)
	{
		pDlg.reset(new ecvProgressDialog(true, parameters.parentWidget)); //cancel available
		pDlg->setMethodTitle(QObject::tr("Save PV file"));
		pDlg->setInfo(QObject::tr("Points: %L1").arg( numberOfPoints ));
		pDlg->start();
	}
	cloudViewer::NormalizedProgress nprogress(pDlg.data(), numberOfPoints);

	CC_FILE_ERROR result = CC_FERR_NO_ERROR;

	for (unsigned i = 0; i < numberOfPoints; i++)
	{
		//write point
		{
			const CCVector3* P = theCloud->getPoint(i);
			
			//conversion to float
			float wBuff[3] = {(float)P->x, (float)P->y, (float)P->z};
			if (out.write((const char*)wBuff,3*sizeof(float)) < 0)
			{
				result = CC_FERR_WRITING;
				break;
			}
		}
			
		//write scalar value
		if (hasSF)
			val = static_cast<float>(theCloud->getPointScalarValue(i));
		if (out.write((const char*)&val,sizeof(float)) < 0)
		{
			result = CC_FERR_WRITING;
			break;
		}

		if (pDlg && !nprogress.oneStep())
		{
			result = CC_FERR_CANCELED_BY_USER;
			break;
		}
	}

	out.close();

	return result;
}

CC_FILE_ERROR PVFilter::loadFile(const QString& filename, ccHObject& container, LoadParameters& parameters)
{
	//opening file
	QFile in(filename);
	if (!in.open(QIODevice::ReadOnly))
		return CC_FERR_READING;

	//we deduce the points number from the file size
	qint64 fileSize = in.size();
	qint64 singlePointSize = 4*sizeof(float);
	//check that size is ok
	if (fileSize == 0)
		return CC_FERR_NO_LOAD;
	if ((fileSize % singlePointSize) != 0)
		return CC_FERR_MALFORMED_FILE;
	unsigned numberOfPoints = static_cast<unsigned>(fileSize  / singlePointSize);

	//progress dialog
	QScopedPointer<ecvProgressDialog> pDlg(0);
	if (parameters.parentWidget)
	{
		pDlg.reset(new ecvProgressDialog(true, parameters.parentWidget)); //cancel available
		pDlg->setMethodTitle(QObject::tr("Open PV file"));
		pDlg->setInfo(QObject::tr("Points: %L1").arg( numberOfPoints ));
		pDlg->start();
	}
	cloudViewer::NormalizedProgress nprogress(pDlg.data(), numberOfPoints);

	ccPointCloud* loadedCloud = 0;
	//if the file is too big, it will be chuncked in multiple parts
	unsigned chunkIndex = 0;
	unsigned fileChunkPos = 0;
	unsigned fileChunkSize = 0;
	//number of points read for the current cloud part
	unsigned pointsRead = 0;
	CC_FILE_ERROR result = CC_FERR_NO_ERROR;

	for (unsigned i = 0; i < numberOfPoints; i++)
	{
		//if we reach the max. cloud size limit, we cerate a new chunk
		if (pointsRead == fileChunkPos+fileChunkSize)
		{
			if (loadedCloud)
			{
				int sfIdx = loadedCloud->getCurrentInScalarFieldIndex();
				if (sfIdx>=0)
				{
					cloudViewer::ScalarField* sf = loadedCloud->getScalarField(sfIdx);
					sf->computeMinAndMax();
					loadedCloud->setCurrentDisplayedScalarField(sfIdx);
					loadedCloud->showSF(true);
				}
				container.addChild(loadedCloud);
			}
			fileChunkPos = pointsRead;
			fileChunkSize = std::min<unsigned>(numberOfPoints - pointsRead, CC_MAX_NUMBER_OF_POINTS_PER_CLOUD);
			loadedCloud = new ccPointCloud(QString("unnamed - Cloud #%1").arg(++chunkIndex));
			if (!loadedCloud || !loadedCloud->reserveThePointsTable(fileChunkSize) || !loadedCloud->enableScalarField())
			{
				result = CC_FERR_NOT_ENOUGH_MEMORY;
				if (loadedCloud)
					delete loadedCloud;
				loadedCloud = 0;
				break;
			}
		}

		//we read the 3 coordinates of the point
		float rBuff[3];
		if (in.read((char*)rBuff, 3 * sizeof(float)) >= 0)
		{
			//conversion to CCVector3
			CCVector3 P((PointCoordinateType)rBuff[0],
						(PointCoordinateType)rBuff[1],
						(PointCoordinateType)rBuff[2]);
			loadedCloud->addPoint(P);
		}
		else
		{
			result = CC_FERR_READING;
			break;
		}

		//then the scalar value
		if (in.read((char*)rBuff, sizeof(float)) >= 0)
		{
			loadedCloud->setPointScalarValue(pointsRead, (ScalarType)rBuff[0]);
		}
		else
		{
			//add fake scalar value for consistency then break
			loadedCloud->setPointScalarValue(pointsRead, 0);
			result = CC_FERR_READING;
			break;
		}

		++pointsRead;

		if (pDlg && !nprogress.oneStep())
		{
			result = CC_FERR_CANCELED_BY_USER;
			break;
		}
	}

	in.close();

	if (loadedCloud)
	{
		loadedCloud->shrinkToFit();
		int sfIdx = loadedCloud->getCurrentInScalarFieldIndex();
		if (sfIdx >= 0)
		{
			cloudViewer::ScalarField* sf = loadedCloud->getScalarField(sfIdx);
			sf->computeMinAndMax();
			loadedCloud->setCurrentDisplayedScalarField(sfIdx);
			loadedCloud->showSF(true);
		}
		container.addChild(loadedCloud);
	}

	return result;
}
