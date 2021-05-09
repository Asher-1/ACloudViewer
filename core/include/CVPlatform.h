//##########################################################################
//#                                                                        #
//#                               cloudViewer                              #
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

#ifndef CV_PLATFORM_HEADER
#define CV_PLATFORM_HEADER

//Defines the following macros (depending on the compilation platform/settings)
//	- CV_WINDOWS / CV_MAC_OS / CV_LINUX
//	- CV_ENV32 / CV_ENV64
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32)
	#define CV_WINDOWS
#if defined(_WIN64)
	#define CV_ENV_64
#else
	#define CV_ENV_32
#endif
#else
#if defined(__APPLE__)
	#define CV_MAC_OS
#else
	#define CV_LINUX
#endif
#if defined(__x86_64__) || defined(__ppc64__)
	#define CV_ENV_64
#else
	#define CV_ENV_32
#endif
#endif


#endif //CV_PLATFORM_HEADER
