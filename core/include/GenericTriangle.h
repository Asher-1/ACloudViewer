//##########################################################################
//#                                                                        #
//#                               CVCoreLib                                #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the  #
//#  License.                                                              #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef GENERIC_TRIANGLE_HEADER
#define GENERIC_TRIANGLE_HEADER

//Local
#include "CVGeom.h"

namespace cloudViewer
{

//! A generic triangle interface
/** Returns (temporary) references to each vertex.
**/
class CV_CORE_LIB_API GenericTriangle
{
public:

	//! Default destructor
	virtual ~GenericTriangle() = default;

	//! Returns the first vertex (A)
	virtual const CCVector3* _getA() const = 0;

	//! Returns the second vertex (B)
	virtual const CCVector3* _getB() const = 0;

	//! Returns the third vertex (C)
	virtual const CCVector3* _getC() const = 0;
};

}

#endif //GENERIC_TRIANGLE_HEADER
