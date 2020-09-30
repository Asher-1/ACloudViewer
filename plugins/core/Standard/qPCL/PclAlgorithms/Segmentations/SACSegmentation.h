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
#ifndef Q_PCL_PLUGIN_SACSEGMENTATION_HEADER
#define Q_PCL_PLUGIN_SACSEGMENTATION_HEADER

#include "BasePclModule.h"

#include <PclUtils/PCLCloud.h>

// QT
#include <QString>

class SACSegmentationDlg;

//! SIFT keypoints extraction
class SACSegmentation : public BasePclModule
{
public:
	SACSegmentation();
	virtual ~SACSegmentation();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	int extractRecursive(
		PointCloudT::Ptr xyzCloud,
		PointCloudT::Ptr cloudRemained,
		std::vector<PointCloudT::Ptr> &cloudExtractions,
		bool recursive/* = false*/);

	SACSegmentationDlg* m_dialog;

	QString m_selectedModel;
	QString m_selectedMethod;

	int m_maxIterations;
	float m_probability;
	float m_minRadiusLimits;
	float m_maxRadiusLimits;
	float m_distanceThreshold;
	int m_methodType;
	int m_modelType;

	bool m_exportExtraction;
	bool m_exportRemaining;

	bool m_useVoxelGrid;
	float m_leafSize;
	bool m_recursiveMode;
	float m_maxRemainingRatio;

	float m_normalDisWeight;
};

#endif // Q_PCL_PLUGIN_SACSEGMENTATION_HEADER
