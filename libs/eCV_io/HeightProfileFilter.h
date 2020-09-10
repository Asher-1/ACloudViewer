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
//#                       COPYRIGHT: SAGE INGENIERIE                       #
//#                                                                        #
//##########################################################################

#ifndef ECV_HEIGHT_PROFILE_HEADER
#define ECV_HEIGHT_PROFILE_HEADER

#include "FileIOFilter.h"

//! Polyline height profile I/O filter
/** This file format contains a 2D series: (curvilinear absisca ; height)
**/
class ECV_IO_LIB_API HeightProfileFilter : public FileIOFilter
{
public:
	HeightProfileFilter();

	//inherited from FileIOFilter
	virtual bool canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const override;
	virtual CC_FILE_ERROR saveToFile(ccHObject* entity, const QString& filename, const SaveParameters& parameters) override;
};

#endif // ECV_HEIGHT_PROFILE_HEADER
