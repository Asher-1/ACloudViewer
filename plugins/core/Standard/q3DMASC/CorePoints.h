#pragma once

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

//qCC_db
#include <ecvPointCloud.h>

//CCLib
#include <ReferenceCloud.h>
#include <GenericProgressCallback.h>

//Qt
#include <QSharedPointer>

namespace masc
{
	//! Core points descriptor
	struct CorePoints
	{
		//! Origin cloud
		ccPointCloud* origin = nullptr;

		//! Core points cloud
		ccPointCloud* cloud = nullptr;

		//! Core points 'role'
		QString role;

		//! Return the size
		inline unsigned size() const { return (cloud ? cloud->size() : 0); }
		//! Return the point index
		inline unsigned originIndex(unsigned i) const { return selection ? selection->getPointGlobalIndex(i) : i; }

		//selection (if any)
		QSharedPointer<cloudViewer::ReferenceCloud> selection;
		enum SubSamplingMethod { NONE, RANDOM, SPATIAL };
		SubSamplingMethod selectionMethod = NONE;
		double selectionParam = std::numeric_limits<double>::quiet_NaN();

		//! Prepares the selection (must be called once)
		bool prepare(cloudViewer::GenericProgressCallback* progressCb = nullptr);
	};

}; //namespace masc
