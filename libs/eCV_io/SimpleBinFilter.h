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
//#                        COPYRIGHT: CNRS / OSUR                          #
//#                                                                        #
//##########################################################################

#ifndef ECV_SIMPLE_BIN_FILTER_HEADER
#define ECV_SIMPLE_BIN_FILTER_HEADER

#include "FileIOFilter.h"

//! Simple binary file (with attached text meta-file)
class ECV_IO_LIB_API SimpleBinFilter : public FileIOFilter
{
public:
	SimpleBinFilter();

	//inherited from FileIOFilter
	virtual CC_FILE_ERROR loadFile(const QString& filename, ccHObject& container, LoadParameters& parameters) override;
	
	virtual bool canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const override;
	virtual CC_FILE_ERROR saveToFile(ccHObject* entity, const QString& filename, const SaveParameters& parameters) override;

protected:

};

#endif // ECV_SIMPLE_BIN_FILTER_HEADER