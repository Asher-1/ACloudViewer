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

#ifndef QPCL_SINGLETON_HEADER
#define QPCL_SINGLETON_HEADER


#include <boost/shared_ptr.hpp>
template<class T> struct ecvBoostSingleton
{
	//! Default constructor
	ecvBoostSingleton() : instance(nullptr) {}
	//! Current instance
	boost::shared_ptr<T> instance;
	//! Destructor
	~ecvBoostSingleton() = default;
	//! Releases the current instance
};

#endif // QPCL_SINGLETON_HEADER
