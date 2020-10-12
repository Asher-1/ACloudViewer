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

#ifndef ECV_GENERIC_VISUALIZER_2D_HEADER
#define ECV_GENERIC_VISUALIZER_2D_HEADER

#include "ecvGenericVisualizer.h"

//! Generic visualizer 2D interface
class ECV_DB_LIB_API ecvGenericVisualizer2D : public ecvGenericVisualizer
{
	Q_OBJECT

public:

	//! Default constructor
	/** \param name object name
	**/
	ecvGenericVisualizer2D() = default;

	//! Destructor
	virtual ~ecvGenericVisualizer2D() = default;
};

#endif // ECV_GENERIC_VISUALIZER_2D_HEADER
