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
#ifndef Q_PCL_PLUGIN_MINIMUMCUT_HEADER
#define Q_PCL_PLUGIN_MINIMUMCUT_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class MinimumCutSegmentationDlg;

//! Region Growing Segmentation
class MinimumCutSegmentation : public BasePclModule
{
public:
	MinimumCutSegmentation();
	virtual ~MinimumCutSegmentation();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	MinimumCutSegmentationDlg* m_dialog;

	bool m_colored;

	int m_neighboursNumber;
	float m_smoothSigma;
	float m_backWeightRadius;
	float m_foregroundWeight;

	float m_cx;
	float m_cy;
	float m_cz;
};

#endif // Q_PCL_PLUGIN_MINIMUMCUT_HEADER
