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
//#                    COPYRIGHT: CLOUDVIEWER  project                     #
//#                                                                        #
//##########################################################################

#ifndef ECV_PCD_FILTER_HEADER
#define ECV_PCD_FILTER_HEADER

// ECV_DB_LIB
#include <FileIOFilter.h>

//! PCD point cloud I/O filter
class PcdFilter : public FileIOFilter
{
public:
	PcdFilter();

	//inherited from FileIOFilter
	virtual CC_FILE_ERROR loadFile(const QString& filename, ccHObject& container, LoadParameters& parameters);
	
	virtual bool canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const;
	virtual CC_FILE_ERROR saveToFile(ccHObject* entity, const QString& filename, const SaveParameters& parameters);

};

#endif // ECV_PCD_FILTER_HEADER
