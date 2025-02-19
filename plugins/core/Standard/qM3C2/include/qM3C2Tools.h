//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER  PLUGIN: qM3C2                       #
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
//#            COPYRIGHT: UNIVERSITE EUROPEENNE DE BRETAGNE                #
//#                                                                        #
//##########################################################################

#ifndef Q_M3C2_TOOLS_HEADER
#define Q_M3C2_TOOLS_HEADER

// CV_CORE_LIB
#include <GenericIndexedCloud.h>
#include <GenericProgressCallback.h>
#include <DgmOctree.h>

class ccGenericPointCloud;
class NormsIndexesTableType;
class ccScalarField;
class ccPointCloud;
class ecvMainAppInterface;

//! M3C2 normals computation related tools
class qM3C2Normals
{
public:

	//! Normals computation mode
	/** \warning Don't change the associated values! (for parameter files)
	**/
	enum ComputationMode
	{
		DEFAULT_MODE			= 0, //compute normals on core points
		USE_CLOUD1_NORMALS		= 1,
		MULTI_SCALE_MODE		= 2,
		VERT_MODE				= 3,
		HORIZ_MODE				= 4,
		USE_CORE_POINTS_NORMALS	= 5,
	};

	//! Computes normals on core points only
	/** See qCC's ccNormalVectors::ComputeCloudNormals.
		\warning normals orientation is not resolved!
	**/
	static bool ComputeCorePointsNormals(	cloudViewer::GenericIndexedCloud* corePoints,
                                                NormsIndexesTableType* corePointsNormals,
                                                ccGenericPointCloud* sourceCloud,
                                                const std::vector<PointCoordinateType>& sortedRadii,
                                                bool& invalidNormals,
                                                int maxThreadCount = 0,
                                                ccScalarField* normalScale = nullptr,
                                                cloudViewer::GenericProgressCallback* progressCb = nullptr,
                                                cloudViewer::DgmOctree* inputOctree = nullptr);
	
	//! Re-orients normal vectors so that they all 'look' towards the nearest point of another cloud
	static bool UpdateNormalOrientationsWithCloud(	cloudViewer::GenericIndexedCloud* normCloud,
                                                        NormsIndexesTableType& normsCodes,
                                                        cloudViewer::GenericIndexedCloud* orientationCloud,
                                                        int maxThreadCount = 0,
                                                        cloudViewer::GenericProgressCallback* progressCb = nullptr);

	//! Makes all normals horizontal
	static void MakeNormalsHorizontal(NormsIndexesTableType& normsCodes);
};

//! M3C2 generic tools
class qM3C2Tools
{
public:

	//! Computes statistics on a neighbors set
	/** Either the mean distance and std. dev. (if useMedian is false)
		or the median and interquartile range (if useMedian is true).
		See http://en.wikipedia.org/wiki/Interquartile_range
	**/
	static void ComputeStatistics(	cloudViewer::DgmOctree::NeighboursSet& set,
                                        bool useMedian,
                                        double& meanOrMedian,
                                        double& stdDevOrIQR);

	//! M3C2 parameters that can be guessed automatically by 'probing'
	struct GuessedParams
	{
		int preferredDimension = -1;
		double normScale = 0.0;
		double projScale = 0.0;
		double projDepth = 0.0;
	};

	//! Tries to guess some M3C2 parameters by randomly 'probing' the cloud
	static bool GuessBestParams(ccPointCloud* cloud1,
                                    ccPointCloud* cloud2,
                                    unsigned minPoints4Stats,
                                    GuessedParams& params,
                                    bool fastMode,
                                    ecvMainAppInterface* app = nullptr,
                                    unsigned probingCount = 1000);
};

#endif // Q_M3C2_TOOLS_HEADER
