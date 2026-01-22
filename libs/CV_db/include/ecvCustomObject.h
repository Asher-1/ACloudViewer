// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvHObject.h"

//! Custom hierarchy object
/** Used internally for deserialization of plugin-defined hierarchy objects
        (see CV_TYPES::CUSTOM_H_OBJECT).
**/
class CV_DB_LIB_API ccCustomHObject : public ccHObject {
public:
    //! Default constructor
    /** \param name object name (optional)
     **/
    ccCustomHObject(QString name = QString()) : ccHObject(name) {}

    // inherited from ccHObject
    bool isSerializable() const override { return true; }

    // inherited from ccObject
    CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::CUSTOM_H_OBJECT;
    }

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
class CV_DB_LIB_API ccCustomLeafObject : public ccCustomHObject {
public:
    //! Default constructor
    /** \param name object name (optional)
     **/
    ccCustomLeafObject(QString name = QString()) : ccCustomHObject(name) {}

    // inherited from ccCustomHObject
    CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::CUSTOM_LEAF_OBJECT;
    }
};
