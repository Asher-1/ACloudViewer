//##########################################################################
//#                                                                        #
//#                       CLOUDVIEWER PLUGIN: qPCL                         #
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
//#                         COPYRIGHT: Asher                               #
//#                                                                        #
//##########################################################################
//
#ifndef Q_PCL_PLUGIN_MARCHINGCUBE_HEADER
#define Q_PCL_PLUGIN_MARCHINGCUBE_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class MarchingCubeDlg;

//! Pcl Grid Projection
class MarchingCubeReconstruction : public BasePclModule
{
public:
	MarchingCubeReconstruction();
	virtual ~MarchingCubeReconstruction();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	MarchingCubeDlg* m_dialog;
	int m_marchingMethod;
	float m_epsilon;
	float m_isoLevel;
	float m_gridResolution;
	float m_percentageExtendGrid;

	int m_knn_radius;
	float m_normalSearchRadius;
	bool m_useKnn;

};

#endif // Q_PCL_PLUGIN_MARCHINGCUBE_HEADER
