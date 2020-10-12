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

#ifndef ECV_CUSTOM_OBJECT_HEADER
#define ECV_CUSTOM_OBJECT_HEADER

//Local
#include "ecvHObject.h"

//! Custom hierarchy object
/** Used internally for deserialization of plugin-defined hierarchy objects
	(see CV_TYPES::CUSTOM_H_OBJECT).
**/
class ECV_DB_LIB_API ccCustomHObject : public ccHObject
{
public:

	//! Default constructor
	/** \param name object name (optional)
	**/
	ccCustomHObject(QString name = QString())
		: ccHObject(name)
	{}

	//inherited from ccHObject
	bool isSerializable() const override { return true; }

	// inherited from ccObject
	CV_CLASS_ENUM getClassID() const override { return CV_TYPES::CUSTOM_H_OBJECT; }

	//! Returns the default key for the "class name" metadata
	/** See ccHObject::New.
	**/
	static QString DefautMetaDataClassName() { return QString("class_name"); }
	//! Returns the default key for the "plugin name" metadata
	/** See ccHObject::New.
	**/
	static QString DefautMetaDataPluginName() { return QString("plugin_name"); }
};

//! Custom leaf object
/** Used internally for deserialization of plugin-defined leaf objects
	(see CV_TYPES::CUSTOM_LEAF_OBJECT).
**/
class ECV_DB_LIB_API ccCustomLeafObject : public ccCustomHObject
{
public:

	//! Default constructor
	/** \param name object name (optional)
	**/
	ccCustomLeafObject(QString name = QString()) : ccCustomHObject(name) {}

	// inherited from ccCustomHObject
	CV_CLASS_ENUM getClassID() const override { return CV_TYPES::CUSTOM_LEAF_OBJECT; }
};

#endif //CC_CUSTOM_OBJECT_HEADER
