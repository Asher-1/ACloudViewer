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

#include "CorePoints.h"

//qCC_db
#include <ecvPointCloud.h>

//CCLib
#include <CloudSamplingTools.h>

//system
#include <assert.h>

using namespace masc;

bool CorePoints::prepare(cloudViewer::GenericProgressCallback* progressCb/*=nullptr*/)
{
	if (!origin)
	{
		assert(false);
		return false;
	}

	if (selection)
	{
		//nothing to do
		return true;
	}
	
	//now we can compute the subsampled version
	cloudViewer::ReferenceCloud* ref = nullptr;
	switch (selectionMethod)
	{
	case SPATIAL:
	{
		//we'll need an octree
		if (!origin->getOctree())
		{
			if (!origin->computeOctree(progressCb))
			{
				CVLog::Warning("[CorePoints::prepare] Failed to compute the octree");
				return false;
			}
		}

		cloudViewer::CloudSamplingTools::SFModulationParams modParams;
		modParams.enabled = false;
		ref = cloudViewer::CloudSamplingTools::resampleCloudSpatially(
			origin,
			static_cast<PointCoordinateType>(selectionParam),
			modParams,
			origin->getOctree().data(),
			progressCb);

		break;
	}

	case RANDOM:
	{
		if (selectionParam <= 0.0 || selectionParam >= 1.0)
		{
			CVLog::Warning("[CorePoints::prepare] Random subsampling ration must be between 0 and 1 (excluded)");
			return false;
		}
		int targetCount = static_cast<int>(origin->size() * selectionParam);
		ref = cloudViewer::CloudSamplingTools::subsampleCloudRandomly(origin, targetCount, progressCb);
		break;
	}

	case NONE:
		//nothing to do
		cloud = origin;
		return true;

	default:
		assert(false);
		break;
	}

	//store the references
	if (!ref)
	{
		CVLog::Warning("[CorePoints::prepare] Failed to subsampled the origin cloud");
		return false;
	}
	selection.reset(ref);

	//and create the subsampled version of the cloud
	cloud = origin->partialClone(ref);
	if (!cloud)
	{
		CVLog::Warning("[CorePoints::prepare] Failed to subsampled the origin cloud (not enough memory)");
		return false;
	}

	return true;
}
