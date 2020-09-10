//##########################################################################
//#                                                                        #
//#                               ECV_IO                                   #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the License.  #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

//#ifndef ECV_IO_HEADER
//#define ECV_IO_HEADER
//
#include <QtCore/QtGlobal>
//
//#if defined( ECV_IO_LIBRARY_BUILD )
//#  define ECV_IO_LIB_API Q_DECL_EXPORT
//#else
//#  define ECV_IO_LIB_API Q_DECL_IMPORT
//#endif
//
//#endif // ECV_IO_HEADER

#ifdef ECV_IO_LIBRARY_BUILD

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the ECV_IO_LIB_EXPORTS
// symbol defined on the command line. this symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// QCC_IO_LIB_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef ECV_IO_LIB_EXPORTS
#define ECV_IO_LIB_API __declspec(dllexport)
#else
#define ECV_IO_LIB_API __declspec(dllimport)
#endif

#else //NOT ECV_IO_LIBRARY_BUILD

#define ECV_IO_LIB_API

#endif //ECV_IO_LIBRARY_BUILD
