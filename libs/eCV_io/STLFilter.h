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

#ifndef ECV_STL_FILTER_HEADER
#define ECV_STL_FILTER_HEADER

#include "FileIOFilter.h"

class ccMesh;
class ccPointCloud;
class ccGenericMesh;

//! StereoLithography file I/O filter
/** See http://www.ennex.com/~fabbers/StL.asp
**/
class ECV_IO_LIB_API STLFilter : public FileIOFilter
{
public:
	STLFilter();

	//inherited from FileIOFilter
	virtual CC_FILE_ERROR loadFile(const QString& filename, ccHObject& container, LoadParameters& parameters) override;
	
	virtual bool canSave(CV_CLASS_ENUM type, bool& multiple, bool& exclusive) const override;
	virtual CC_FILE_ERROR saveToFile(ccHObject* entity, const QString& filename, const SaveParameters& parameters) override;

protected:

	//! Custom save method
	CC_FILE_ERROR saveToASCIIFile(ccGenericMesh* mesh, FILE *theFile, QWidget* parentWidget = 0);
	CC_FILE_ERROR saveToBINFile(ccGenericMesh* mesh, FILE *theFile, QWidget* parentWidget = 0);

	//! Custom load method for ASCII files
	CC_FILE_ERROR loadASCIIFile(QFile& fp,
								ccMesh* mesh,
								ccPointCloud* vertices,
								LoadParameters& parameters);

	//! Custom load method for binary files
	CC_FILE_ERROR loadBinaryFile(QFile& fp,
								ccMesh* mesh,
								ccPointCloud* vertices,
								LoadParameters& parameters);
};

#endif // ECV_STL_FILTER_HEADER
