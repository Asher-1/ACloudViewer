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

#ifndef ECV_REGISTRATION_TOOLS_HEADER
#define ECV_REGISTRATION_TOOLS_HEADER

//CVLib
#include <RegistrationTools.h>

//ECV_DB_LIB
#include <ecvGLMatrix.h>

class QWidget;
class QStringList;
class ccHObject;

//! Registration tools wrapper
class ccRegistrationTools
{

public:

	//! Applies ICP registration on two entities
	/** \warning Automatically samples points on meshes if necessary (see code for magic numbers ;)
	**/
	static bool ICP(ccHObject* data,
					ccHObject* model,
					ccGLMatrix& transMat,
					double& finalScale,
					double& finalRMS,
					unsigned& finalPointCount,
					double minRMSDecrease,
					unsigned maxIterationCount,
					unsigned randomSamplingLimit,
					bool removeFarthestPoints,
					CVLib::ICPRegistrationTools::CONVERGENCE_TYPE method,
					bool adjustScale,
					double finalOverlapRatio = 1.0,
					bool useDataSFAsWeights = false,
					bool useModelSFAsWeights = false,
					int transformationFilters = CVLib::ICPRegistrationTools::SKIP_NONE,
					int maxThreadCount = 0,
					QWidget* parent = nullptr);

};

#endif // ECV_REGISTRATION_TOOLS_HEADER