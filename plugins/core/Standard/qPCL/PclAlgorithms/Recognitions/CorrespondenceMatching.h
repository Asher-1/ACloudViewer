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

#ifndef Q_PCL_PLUGIN_CORRESPONDENCEMATCHING_HEADER
#define Q_PCL_PLUGIN_CORRESPONDENCEMATCHING_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class CorrespondenceMatchingDialog;

//! Correspondence Matching
class CorrespondenceMatching : public BasePclModule
{
public:
	CorrespondenceMatching();
	virtual ~CorrespondenceMatching();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	void CorrespondenceMatching::applyTransformation(
		ccHObject* entity, const ccGLMatrixd& mat);

	ccPointCloud* m_sceneCloud;
	CorrespondenceMatchingDialog* m_dialog;

	float m_leafSize;
	bool m_useVoxelGrid;
	bool m_gcMode;
	bool m_verification;
	int m_maxThreadCount;

	float m_modelSearchRadius;
	float m_sceneSearchRadius;
	float m_shotDescriptorRadius;
	int m_normalKSearch;

	// GC
	float m_consensusResolution;
	float m_gcMinClusterSize;

	// Hough
	float m_lrfRadius;
	float m_houghBinSize;
	float m_houghThreshold;

	// ICP
	int m_icpMaxIterations;
	float m_icpCorrDistance;

	// Global Hypotheses Verification
	float clusterReg = 5.0f;
	float inlierThreshold = 0.005f;
	float occlusionThreshold = 0.01f;
	float radiusClutter = 0.03f;
	float regularizer = 3.0f;
	float radiusNormals = 0.05f;
	bool detectClutter = true;

	std::vector< float > m_scales;
	std::vector< ccPointCloud* > m_modelClouds;
};

#endif // Q_PCL_PLUGIN_CORRESPONDENCEMATCHING_HEADER
