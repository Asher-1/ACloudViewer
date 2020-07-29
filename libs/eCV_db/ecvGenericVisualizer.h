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

#ifndef ECV_GENERIC_VISUALIZER_HEADER
#define ECV_GENERIC_VISUALIZER_HEADER

// LOCAL
#include "eCV_db.h"

// CV_CORE_LIB
#include <CVLog.h>
#include <CVGeom.h>

// QT
#include <QObject>

//! Generic visualizer interface
class ECV_DB_LIB_API ecvGenericVisualizer : public QObject
{
	Q_OBJECT

public:

	//! Default constructor
	/** \param name object name
	**/
	ecvGenericVisualizer() = default;

	//! Destructor
	virtual ~ecvGenericVisualizer() = default;
};

#endif // ECV_GENERIC_VISUALIZER_HEADER
