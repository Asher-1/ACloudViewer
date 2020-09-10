//##########################################################################
//#                                                                        #
//#                               CVLIB                                    #
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

#ifndef CV_SHAREABLE_HEADER
#define CV_SHAREABLE_HEADER

//Local
#include "CVCoreLib.h"

//system
#include <cassert>

//Activate shared objects tracking (for debug purposes)
#ifdef CV_DEBUG
//#define CV_TRACK_ALIVE_SHARED_OBJECTS
#endif

//CCShareable object (with counter)
class CV_CORE_LIB_API CCShareable
{
public:

	//! Default constructor
	CCShareable();

	//! Increase counter
	/** Should be called when this object is 'attached' to another one.
	**/
	virtual void link();

	//! Decrease counter and deletes object when 0
	/** Should be called when this object is 'detached'.
	**/
	virtual void release();

	//! Returns the current link count
	/** \return current link count
	**/
	inline virtual unsigned getLinkCount() const { return m_linkCount; }

#ifdef CV_TRACK_ALIVE_SHARED_OBJECTS
	//! Get alive shared objects count
	static unsigned GetAliveCount();
#endif

protected:

	//! Destructor
	/** private to avoid deletion with 'delete' operator
	**/
	virtual ~CCShareable();

	//! Links counter
	unsigned m_linkCount;
};

#endif // CV_SHAREABLE_HEADER
