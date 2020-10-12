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
//#          COPYRIGHT: EDF R&D / TELECOM ParisTech (ENST-TSI)             #
//#                                                                        #
//##########################################################################

#ifndef ECV_MA_FILTER_HEADER
#define ECV_MA_FILTER_HEADER

#include "FileIOFilter.h"

//! Maya ASCII meshes file I/O filter
class MAFilter : public FileIOFilter
{
public:
	MAFilter();

	//inherited from FileIOFilter
	virtual bool canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const override;
	virtual CC_FILE_ERROR saveToFile(ccHObject* entity, const QString& filename, const SaveParameters& parameters) override;

};

#endif // ECV_MA_FILTER_HEADER
