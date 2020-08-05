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

#ifndef ECV_PDMS_FILTER_HEADER
#define ECV_PDMS_FILTER_HEADER

#ifdef CV_PDMS_SUPPORT

#include "../FileIOFilter.h"

//! PDMS .mac file I/O filter
class ECV_IO_LIB_API PDMSFilter : public FileIOFilter
{
public:
	PDMSFilter();

	//inherited from FileIOFilter
	virtual CC_FILE_ERROR loadFile(const QString& filename, ccHObject& container, LoadParameters& parameters) override;

};

#endif // ECV_PDMS_SUPPORT

#endif // ECV_PDMS_FILTER_HEADER
