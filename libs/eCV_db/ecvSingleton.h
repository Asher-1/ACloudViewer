//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                              #
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

#ifndef ECV_SINGLETON_HEADER
#define ECV_SINGLETON_HEADER


//! Generic singleton encapsulation structure
template<class T> struct ecvSingleton
{
	//! Default constructor
	ecvSingleton() : instance(nullptr) {}
	//! Destructor
	~ecvSingleton() { release(); }
	//! Releases the current instance
	inline void release() { if (instance) { delete instance; instance = nullptr; } }
	
	//! Current instance
	T* instance;
};
#endif //ECV_SINGLETON_HEADER
