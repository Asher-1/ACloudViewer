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

#ifndef ECV_BIN_FILTER_HEADER
#define ECV_BIN_FILTER_HEADER

#include "FileIOFilter.h"


//! CLOUDVIEWER  dedicated binary point cloud I/O filter
class ECV_IO_LIB_API BinFilter : public FileIOFilter
{
public:
	BinFilter();

	//static accessors
	static inline QString GetFileFilter() { return "CloudViewer entities (*.bin)"; }
	static inline QString GetDefaultExtension() { return "bin"; }

	//inherited from FileIOFilter
	virtual CC_FILE_ERROR loadFile(const QString& filename, ccHObject& container, LoadParameters& parameters) override;
	
	virtual bool canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const override;
	virtual CC_FILE_ERROR saveToFile(ccHObject* entity, const QString& filename, const SaveParameters& parameters) override;

	//! old style BIN loading
	static CC_FILE_ERROR LoadFileV1(QFile& in, ccHObject& container, unsigned nbScansTotal, const LoadParameters& parameters);

	//! new style BIN loading
	static CC_FILE_ERROR LoadFileV2(QFile& in, ccHObject& container, int flags);

	//! new style BIN saving
	static CC_FILE_ERROR SaveFileV2(QFile& out, ccHObject* object);

};

#endif //CC_BIN_FILTER_HEADER
