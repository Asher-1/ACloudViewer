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

#include "ScalarFieldCollector.h"

//qCC_db
#include <ecvPointCloud.h>

//CCLib
#include <ScalarField.h>

//system
#include <assert.h>

void SFCollector::push(ccPointCloud* cloud, cloudViewer::ScalarField* sf, Behavior behavior)
{
	assert(!scalarFields.contains(sf));
//	if (scalarFields.contains(sf))
//		CVLog::Warning(QString("[SFCollector] scalar field '%1' HAS ALREADY BEEN COLLECTED").arg(sf->getName()) + ", behaviour " + QString::number(behavior));
//	else
//		CVLog::Warning(QString("[SFCollector] collect scalar field '%1'").arg(sf->getName()) + ", behaviour " + QString::number(behavior));
	SFDesc desc;
	desc.behavior = behavior;
	desc.cloud = cloud;
	scalarFields[sf] = desc;
}

void SFCollector::releaseSFs(bool keepByDefault)
{
	for (Map::iterator it = scalarFields.begin(); it != scalarFields.end(); ++it)
	{
		const SFDesc& desc = it.value();
		cloudViewer::ScalarField* sf = it.key();

		if (desc.behavior == ALWAYS_KEEP || (keepByDefault && desc.behavior == CAN_REMOVE))
		{
			// CVLog::Warning(QString("[SFCollector] Keep scalar field '%1' on cloud '%2'").arg(sf->getName()).arg(desc.cloud->getName()));
			//keep this SF
			continue;
		}
		
		int sfIdx = desc.cloud->getScalarFieldIndexByName(sf->getName());
		if (sfIdx >= 0)
		{
			// CVLog::Warning(QString("[SFCollector] Remove scalar field '%1' from '%2'").arg(sf->getName()).arg(desc.cloud->getName()));
			desc.cloud->deleteScalarField(sfIdx);
		}
		else
		{
			// CVLog::Warning(QString("[SFCollector] Scalar field '%1' can't be found anymore on cloud '%2', impossible to remove it").arg(sf->getName()).arg(desc.cloud->getName()));
		}
	}

	scalarFields.clear();
}

bool SFCollector::setBehavior(cloudViewer::ScalarField *sf, Behavior behavior)
{
	if (scalarFields.contains(sf))
	{
//		Behavior previousBehavior = scalarFields[sf].behavior;
		scalarFields[sf].behavior = behavior;
//		CVLog::Warning("behavior of " + QString(sf->getName()) + " changed from " + QString::number(previousBehavior) + " to " + QString::number(behavior));
	}

	return true;
}
