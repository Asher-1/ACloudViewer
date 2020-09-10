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
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//
#ifndef Q_PCL_PLUGIN_PROJECTIONFILTER_HEADER
#define Q_PCL_PLUGIN_PROJECTIONFILTER_HEADER

#include "BasePclModule.h"

class ProjectionFilterDlg;

//! Projection Filter
class ProjectionFilter : public BasePclModule
{
public:
	ProjectionFilter();
	virtual ~ProjectionFilter();

	//inherited from BasePclModule
	virtual int compute();

protected:

	//inherited from BasePclModule
	virtual int openInputDialog();
	virtual int checkParameters();
	virtual void getParametersFromDialog();
	virtual QString getErrorMessage(int errorCode);

	ProjectionFilterDlg* m_dialog;

	bool m_projectionMode;

	// projection parameters
	float m_coefficientA;
	float m_coefficientB;
	float m_coefficientC;
	float m_coefficientD;

	// boundary parameters
	bool m_useVoxelGrid;
	float m_leafSize;
	bool m_useKnn;
	int m_knn_radius;
	float m_normalSearchRadius;
	float m_boundarySearchRadius;
	float m_boundaryAngleThreshold;

};

#endif // Q_PCL_PLUGIN_PROJECTIONFILTER_HEADER
