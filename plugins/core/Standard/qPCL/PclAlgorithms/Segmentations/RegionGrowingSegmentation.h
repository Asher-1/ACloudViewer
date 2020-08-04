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
#ifndef Q_PCL_PLUGIN_REGIONGROWING_HEADER
#define Q_PCL_PLUGIN_REGIONGROWING_HEADER

#include "BasePclModule.h"

//Qt
#include <QString>

class RegionGrowingSegmentationDlg;

//! Region Growing Segmentation
class RegionGrowingSegmentation : public BasePclModule
{
public:
	RegionGrowingSegmentation();
	virtual ~RegionGrowingSegmentation();

	//inherited from BasePclModule
	virtual int compute();

protected:
	//inherited from BasePclModule
	virtual int checkSelected();
	virtual int openInputDialog();
	virtual void getParametersFromDialog();
	virtual int checkParameters();
	virtual QString getErrorMessage(int errorCode);

	RegionGrowingSegmentationDlg* m_dialog;

	bool m_basedRgb;

	// Basic Region Growing Segmentation Parameters
	int m_k_search;
	int m_min_cluster_size;
	int m_max_cluster_size;
	unsigned int m_neighbour_number;
	float m_smoothness_theta;
	float m_curvature;

	// Region Growing Segmentation Based on RGB Parameters
	int m_min_cluster_size_rgb;
	float m_neighbors_distance;
	float m_point_color_diff;
	float m_region_color_diff;
};

#endif // Q_PCL_PLUGIN_REGIONGROWING_HEADER
