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
#ifndef Q_PCL_PLUGIN_POISSONRECONSTRUCTION_HEADER
#define Q_PCL_PLUGIN_POISSONRECONSTRUCTION_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class PoissonReconstructionDlg;

//! Poisson Reconstruction
class PoissonReconstruction : public BasePclModule
{
public:
	PoissonReconstruction();
	virtual ~PoissonReconstruction();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	PoissonReconstructionDlg* m_dialog;
	int m_degree;
	int m_treeDepth;
	int m_isoDivideDepth;
	int m_solverDivideDepth;
	float m_scale;
	float m_samplesPerNode;

	bool m_useConfidence;
	bool m_useManifold;
	bool m_outputPolygons;

	int m_knn_radius;
	float m_normalSearchRadius;
	bool m_useKnn;

};

#endif // Q_PCL_PLUGIN_POISSONRECONSTRUCTION_HEADER
