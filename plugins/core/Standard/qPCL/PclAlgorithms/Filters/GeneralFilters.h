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
//#                         COPYRIGHT: DAHAI LU                         #
//#                                                                        #
//##########################################################################
//
#ifndef Q_PCL_PLUGIN_GENERALFILTERS_HEADER
#define Q_PCL_PLUGIN_GENERALFILTERS_HEADER

#include "BasePclModule.h"
#include <CVGeom.h>

class ccBBox;
class ccPolyline;
class GeneralFiltersDlg;

//! Projection Filter
class GeneralFilters : public BasePclModule
{
public:
	GeneralFilters();
	virtual ~GeneralFilters();

	//inherited from BasePclModule
	virtual int compute();

protected:

	//inherited from BasePclModule
	virtual int openInputDialog();
	virtual int checkParameters();
	virtual void getParametersFromDialog();
	virtual QString getErrorMessage(int errorCode);

	GeneralFiltersDlg* m_dialog;

	bool m_keepColors;

	bool m_hasColors;

	enum FilterType
	{
		PASS_FILTER,
		CR_FILTER,
		VOXEL_FILTER,
		PM_FILTER,
		HULL_FILTER
	};

	FilterType m_filterType;

	// Condition Removal Parameters
	QString m_comparisonField;
	QStringList m_comparisonTypes;
	float m_minMagnitude;
	float m_maxMagnitude;

	// Progressive Morphological Parameters
	bool m_extractRemainings;
	int m_maxWindowSize;
	float m_slope;
	float m_initialDistance;
	float m_maxDistance;
	
	// Voxel Grid Filter Parameters
	//bool m_sampleAllData;
	float m_leafSize;

	// Crop Hull Parameters
	ccPolyline* m_polyline;
	std::vector<CCVector3> m_boundary;
	int m_dimension;

};

#endif // Q_PCL_PLUGIN_GENERALFILTERS_HEADER
