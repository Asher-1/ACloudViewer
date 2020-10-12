//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
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
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef ECV_PV_FILTER_HEADER
#define ECV_PV_FILTER_HEADER

#include "FileIOFilter.h"

//! Point+Scalar value binary point cloud file I/O filter
class /*QCC_IO_LIB_API*/ PVFilter : public FileIOFilter
{
public:
	PVFilter();

	//inherited from FileIOFilter
	virtual CC_FILE_ERROR loadFile(const QString& filename, ccHObject& container, LoadParameters& parameters) override;
	
	virtual bool canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const override;
	virtual CC_FILE_ERROR saveToFile(ccHObject* entity, const QString& filename, const SaveParameters& parameters) override;

};

#endif // ECV_PV_FILTER_HEADER
