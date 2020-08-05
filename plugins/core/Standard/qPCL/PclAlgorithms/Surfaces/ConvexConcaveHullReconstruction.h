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
#ifndef Q_PCL_PLUGIN_CONVEXCONCAVERECONSTRUCTION_HEADER
#define Q_PCL_PLUGIN_CONVEXCONCAVERECONSTRUCTION_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class ConvexConcaveHullDlg;

//! Convex Concave Hull Reconstruction
class ConvexConcaveHullReconstruction : public BasePclModule
{
public:
	ConvexConcaveHullReconstruction();
	virtual ~ConvexConcaveHullReconstruction();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	ConvexConcaveHullDlg* m_dialog;
	int m_dimension;
	float m_alpha;
};

#endif // Q_PCL_PLUGIN_CONVEXCONCAVERECONSTRUCTION_HEADER
