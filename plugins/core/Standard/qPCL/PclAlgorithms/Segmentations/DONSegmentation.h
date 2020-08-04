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
#ifndef Q_PCL_PLUGIN_DONSEGMENTATION_HEADER
#define Q_PCL_PLUGIN_DONSEGMENTATION_HEADER

#include "BasePclModule.h"

// QT
#include <QString>

class DONSegmentationDlg;

//! SIFT keypoints extraction
class DONSegmentation : public BasePclModule
{
public:
	DONSegmentation();
	virtual ~DONSegmentation();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	DONSegmentationDlg* m_dialog;

	QString m_comparisonField;
	QStringList m_comparisonTypes;
	float m_smallScale;
	float m_largeScale;
	float m_minDonMagnitude;
	float m_maxDonMagnitude;

	int m_minClusterSize;
	int m_maxClusterSize;
	float m_clusterTolerance;
	bool m_randomClusterColor;

};

#endif // Q_PCL_PLUGIN_DONSEGMENTATION_HEADER
