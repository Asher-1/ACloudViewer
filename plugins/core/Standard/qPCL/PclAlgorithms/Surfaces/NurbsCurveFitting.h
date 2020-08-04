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
#ifndef Q_PCL_PLUGIN_NURBSCURVE_HEADER
#define Q_PCL_PLUGIN_NURBSCURVE_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class NurbsCurveFittingDlg;

//! Greedy Triangulation
class NurbsCurveFitting : public BasePclModule
{
public:
	NurbsCurveFitting();
	virtual ~NurbsCurveFitting();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	NurbsCurveFittingDlg* m_dialog;
	bool m_exportProjectedCloud;
	bool m_useVoxelGrid;
	int m_minimizationType;

	bool m_curveFitting3D;
	bool m_closed;

	int m_order;
	int m_curveResolution;
	int m_controlPoints;
	float m_curveSmoothness;
	float m_curveRscale;
};

#endif // Q_PCL_PLUGIN_NURBSCURVE_HEADER
