//##########################################################################
//#                                                                        #
//#                     ACLOUDVIEWER PLUGIN: q3DMASC                       #
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
//#                 COPYRIGHT: Dimitri Lague / CNRS / UEB                  #
//#                                                                        #
//##########################################################################

#include "ContextBasedFeature.h"

//Local
#include "q3DMASCTools.h"

//qCC_db
#include <ecvScalarField.h>

//Qt
#include <QMutex>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace masc;

bool ContextBasedFeature::checkValidity(QString corePointRole, QString &error) const
{
	if (!Feature::checkValidity(corePointRole, error))
	{
		return false;
	}

	if (type == Invalid)
	{
		assert(false);
		error = "invalid feature type";
		return false;
	}

	if (!cloud1)
	{
		error = "one cloud is required to compute context-based features";
		return false;
	}

	cloudViewer::ScalarField* classifSF = Tools::GetClassificationSF(cloud1);
	if (!classifSF)
	{
		error = QString("Context cloud (%1) has no classification field").arg(cloud1Label);
		return false;
	}
	if (classifSF->size() < cloud1->size())
	{
		error = QString("Context cloud (%1) has an invalid classification field").arg(cloud1Label);
		return false;
	}

	if (!scaled() && kNN < 1)
	{
		error = QString("invalid kNN value for a scale-less context-based feature (%1)").arg(kNN);
		return false;
	}

	return true;
}

bool ContextBasedFeature::prepare(	const CorePoints& corePoints,
									QString& errorMessage,
									cloudViewer::GenericProgressCallback* progressCb/*=nullptr*/,
									SFCollector* generatedScalarFields/*=nullptr*/)
{
	if (!corePoints.cloud)
	{
		//invalid input
		assert(false);
		errorMessage = "internal error (no input core points)";
		return false;
	}

	if (!cloud1)
	{
		//invalid input
		assert(false);
		errorMessage = "internal error (no contextual cloud)";
		return false;
	}

	if (!checkValidity(corePoints.role, errorMessage))
	{
		assert(false);
		return false;
	}

	cloudViewer::ScalarField* classifSF = Tools::GetClassificationSF(cloud1);
	if (!classifSF || classifSF->size() < cloud1->size())
	{
		assert(false);
		//already checked by 'checkValidity'
		return false;
	}
	cloud1->setCurrentOutScalarField(cloud1->getScalarFieldIndexByName(classifSF->getName()));

	//build the final SF name
	QString typeStr = ToString(type);
	QString resultSFName = typeStr + "_" + cloud1Label + "_" + QString::number(ctxClassLabel);
	if (scaled())
	{
		resultSFName += "@" + QString::number(scale);
	}
	else
	{
		resultSFName += "@kNN=" + QString::number(kNN);
	}

	//and the scalar field
	assert(!sf);
	sf1WasAlreadyExisting = CheckSFExistence(corePoints.cloud, qPrintable(resultSFName));
	sf = PrepareSF(corePoints.cloud, qPrintable(resultSFName), generatedScalarFields, SFCollector::CAN_REMOVE);
	if (!sf)
	{
		errorMessage = QString("[ContextBasedFeature::prepare] Failed to prepare scalar %1 @ scale %2").arg(resultSFName).arg(scale);
		return false;
	}
	source.name = sf->getName();

	// NOT NECESSARY IF THE VALUE IS ALREADY COMPUTED
	if (!scaled() && !sf1WasAlreadyExisting) //with 'kNN' neighbors, we can compute the values right away
	{
		unsigned pointCount = corePoints.size();
		QString logMessage = "Computing " + typeStr
							 + " on cloud " + corePoints.cloud->getName() + " (" + QString::number(pointCount) + " points)"
							 + " with context cloud " + cloud1Label
							 + " (class " + QString::number(ctxClassLabel) + ")";

		//first: look for the number of points in the relevent class
		const ScalarType fClass = static_cast<ScalarType>(ctxClassLabel);
		unsigned classCount = 0;
		for (unsigned i = 0; i < classifSF->size(); ++i)
		{
			if (classifSF->getValue(i) == fClass)
				++classCount;
		}

		if (classCount >= static_cast<unsigned>(kNN))
		{
			ccPointCloud classCloud;
			if (!classCloud.reserve(classCount))
			{
				errorMessage = "Not enough memory";
				return false;
			}
			
			for (unsigned i = 0; i < classifSF->size(); ++i)
			{
				if (classifSF->getValue(i) == fClass)
				{
					classCloud.addPoint(*cloud1->getPoint(i));
				}
			}

			//compute the octree
			CVLog::Print(QString("Computing octree of class %1 (%2 points)").arg(ctxClassLabel).arg(classCount));
			ccOctree::Shared classOctree = classCloud.computeOctree(progressCb);
			if (!classOctree)
			{
				errorMessage = "[ContextBasedFeature::prepare] Failed to compute octree (not enough memory?)";
				return false;
			}

			//now extract the neighborhoods
			unsigned char octreeLevel = classOctree->findBestLevelForAGivenPopulationPerCell(static_cast<unsigned>(std::max(3, kNN)));
			CVLog::Print(QString("[Initial octree level] level = %1").arg(octreeLevel));

			if (progressCb)
			{
				progressCb->setMethodTitle(qPrintable("Compute " + typeStr));
				progressCb->setInfo(qPrintable(logMessage));
			}
			CVLog::Print(logMessage);
			cloudViewer::NormalizedProgress nProgress(progressCb, pointCount);

			QMutex mutex;
			double meanNeighborhoodSize = 0;
			int tenth = pointCount / 10;
			bool cancelled = false;
#ifndef _DEBUG
#if defined(_OPENMP)
#pragma omp parallel for num_threads(std::max(1, omp_get_max_threads() - 2))
#endif
#endif
			for (int i = 0; i < static_cast<int>(pointCount); ++i)
			{
			if (!cancelled)
			{
				const CCVector3* P = corePoints.cloud->getPoint(i);
				cloudViewer::ReferenceCloud Yk(&classCloud);
				double maxSquareDist = 0;

				ScalarType s = NAN_VALUE;

				int neighborhoodSize = 0;
				if (classOctree->findPointNeighbourhood(P, &Yk, static_cast<unsigned>(kNN), octreeLevel, maxSquareDist, 0, &neighborhoodSize) >= static_cast<unsigned>(kNN))
				{
					CCVector3d sumQ(0, 0, 0);
					for (int k = 0; k < kNN; ++k)
					{
						sumQ += CCVector3d::fromArray(Yk.getPoint(k)->u);
					}

					switch (type)
					{
					case DZ:
						s = static_cast<ScalarType>(P->z - sumQ.z / kNN);
						break;
					case DH:
						s = static_cast<ScalarType>(sqrt(pow(P->x - sumQ.x / kNN, 2.0) + pow(P->y - sumQ.y / kNN, 2.0)));
						break;
					}
				}

				if (i && (i % tenth) == 0)
				{
					double density = meanNeighborhoodSize / tenth;
					if (density < 1.1)
					{
						if (octreeLevel + 1 < cloudViewer::DgmOctree::MAX_OCTREE_LEVEL)
							++octreeLevel;
					}
					else while (density > 2.9)
					{
						if (octreeLevel <= 5)
							break;
						--octreeLevel;
						density /= 2.0;
					}
					CVLog::Print(QString("[Adaptative octree level] Mean neighborhood size: %1 --> new level = %2").arg(meanNeighborhoodSize / tenth).arg(octreeLevel));
					meanNeighborhoodSize = 0;
				}
				else
				{
					meanNeighborhoodSize += neighborhoodSize;
				}

				sf->setValue(i, s);

				if (progressCb)
				{
					mutex.lock();
					cancelled = !nProgress.oneStep();
					mutex.unlock();
					if (cancelled)
					{
						//process cancelled by the user
						errorMessage = "[ContextBasedFeature] Process cancelled";
					}
				}
			}
			}

			if (progressCb)
			{
				progressCb->stop();
			}

			if (cancelled)
			{
				sf->computeMinAndMax();
				return false;
			}

		} // classCount >= kNN
		else // classCount < kNN
		{
			//specific case: not enough points of this class in the whole cloud!
			//sf->fill(NAN_VALUE); //already the case
			CVLog::Warning(QString("Cloud %1 has less than %2 points of class %3").arg(cloud1Label).arg(classCount).arg(ctxClassLabel));
		}

		sf->computeMinAndMax();
	}

	return true;
}

bool ContextBasedFeature::computeValue(cloudViewer::DgmOctree::NeighboursSet& pointsInNeighbourhood, const CCVector3& queryPoint, ScalarType& outputValue) const
{
	const ScalarType fClass = static_cast<ScalarType>(ctxClassLabel);

	CCVector3d sumQ(0, 0, 0);
	unsigned validCount = 0;
	for (cloudViewer::DgmOctree::PointDescriptor& Pd : pointsInNeighbourhood)
	{
		//we only consider points with the right class!!!
		if (cloud1->getPointScalarValue(Pd.pointIndex) != fClass)
			continue;
		sumQ += CCVector3d::fromArray(Pd.point->u);
		++validCount;
	}

	if (validCount == 0)
	{
		outputValue = NAN_VALUE;
		return true;
	}

	switch (type)
	{
	case DZ:
		outputValue = static_cast<ScalarType>(queryPoint.z - sumQ.z / validCount);
		break;
	case DH:
		outputValue = static_cast<ScalarType>(sqrt(pow(queryPoint.x - sumQ.x / validCount, 2.0) + pow(queryPoint.y - sumQ.y / validCount, 2.0)));
		break;
	default:
		assert(false);
		outputValue = NAN_VALUE;
		return false;
	}

	return true;
}


bool ContextBasedFeature::finish(const CorePoints& corePoints, QString& error)
{
	if (!corePoints.cloud)
	{
		//invalid input
		assert(false);
		error = "internal error (no input core points)";
		return false;
	}

	bool success = true;

	if (sf)
	{
		sf->computeMinAndMax();

		//update display
		//if (corePoints.cloud->getDisplay())
		{
			int sfIndex = corePoints.cloud->getScalarFieldIndexByName(sf->getName());
			corePoints.cloud->setCurrentDisplayedScalarField(sfIndex);
			//corePoints.cloud->getDisplay()->redraw();
			//QCoreApplication::processEvents();
		}
	}

	return success;
}

QString ContextBasedFeature::toString() const
{
	//use the default keyword + number of neighbors + the scale + the context class
	QString str = ToString(type);
	if (!scaled())
	{
		if (kNN != 1)
			str += QString::number(kNN);
		str += "_SC0";
	}
	else
	{
		str += "_SC" + QString::number(scale);
	}

	str += "_" + cloud1Label + "_" + QString::number(ctxClassLabel);

	return str;
}
