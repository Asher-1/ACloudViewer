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
#ifndef Q_PCL_PLUGIN_GREEDYTRIANGULATION_HEADER
#define Q_PCL_PLUGIN_GREEDYTRIANGULATION_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class GreedyTriangulationDlg;

//! Greedy Triangulation
class GreedyTriangulation : public BasePclModule
{
public:
	GreedyTriangulation();
	virtual ~GreedyTriangulation();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	GreedyTriangulationDlg* m_dialog;
	int m_trigulationSearchRadius;
	float m_weightingFactor;
	int m_maxNearestNeighbors;
	int m_maxSurfaceAngle;
	int m_minAngle;
	int m_maxAngle;
	bool m_normalConsistency;

	int m_knn_radius;
	float m_normalSearchRadius;
	bool m_useKnn;

};

#endif // Q_PCL_PLUGIN_GREEDYTRIANGULATION_HEADER
