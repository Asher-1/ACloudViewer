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

#ifdef _MSC_VER
//To get rid of the really annoying warnings about template class exportation
#pragma warning( disable: 4251 )
#pragma warning( disable: 4530 )
#endif

#ifdef CV_USE_AS_DLL

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the CV_CORE_LIB_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// CV_CORE_LIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.

//#ifdef CV_CORE_LIB_EXPORTS
//#define CV_CORE_LIB_API __declspec(dllexport)
//#else
//#define CV_CORE_LIB_API __declspec(dllimport)
//#endif

#define CV_CORE_LIB_API __declspec(dllexport)

#else //NOT CV_USE_AS_DLL

#define CV_CORE_LIB_API

#endif //CV_USE_AS_DLL
