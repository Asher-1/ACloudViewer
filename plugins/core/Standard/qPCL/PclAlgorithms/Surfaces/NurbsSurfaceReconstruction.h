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
#ifndef Q_PCL_PLUGIN_NURBSSURFACE_HEADER
#define Q_PCL_PLUGIN_NURBSSURFACE_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class NurbsSurfaceDlg;

//! Greedy Triangulation
class NurbsSurfaceReconstruction : public BasePclModule
{
public:
	NurbsSurfaceReconstruction();
	virtual ~NurbsSurfaceReconstruction();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	NurbsSurfaceDlg* m_dialog;
	bool m_useVoxelGrid;

	int m_order;
	int m_meshResolution;
	int m_curveResolution;
	int m_refinements;
	int m_iterations;
	bool m_twoDim;
	bool m_fitBSplineCurve;

	float m_interiorSmoothness;
	float m_interiorWeight;
	float m_boundarySmoothness;
	float m_boundaryWeight;
};

#endif // Q_PCL_PLUGIN_NURBSSURFACE_HEADER
