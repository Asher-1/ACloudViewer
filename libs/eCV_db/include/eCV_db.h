//##########################################################################
//#                                                                        #
//#                               ECV_DB                                   #
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

#ifndef ECV_DB_HEADER
#define ECV_DB_HEADER

#include <QtCore/QtGlobal>

//#if defined( ECV_DB_LIBRARY_BUILD )
//#  define ECV_DB_LIB_API Q_DECL_EXPORT
//#else
//#  define ECV_DB_LIB_API Q_DECL_IMPORT
//#endif
//#endif // ECV_DB_HEADER

#ifdef ECV_DB_LIBRARY_BUILD

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the ECV_DB_LIB_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// ECV_DB_LIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef ECV_DB_LIB_EXPORTS

#define ECV_DB_LIB_API __declspec(dllexport)
#else //NOT ECV_DB_LIB_EXPORTS
#define ECV_DB_LIB_API __declspec(dllimport)
#endif //NOT ECV_DB_LIB_EXPORTS

#else //NOT ECV_DB_LIBRARY_BUILD

#define ECV_DB_LIB_API

#endif // NOT ECV_DB_LIBRARY_BUILD

#endif // ECV_DB_HEADER