// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "CVShareable.h"
#include "ecvHObject.h"
#include "ecvMaterial.h"

//! Mesh (triangle) material
class CV_DB_LIB_API ccMaterialSet : public std::vector<ccMaterial::CShared>,
                                    public CCShareable,
                                    public ccHObject {
public:
    //! Default constructor
    ccMaterialSet(const QString& name = QString());

    // inherited from ccHObject
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::MATERIAL_SET;
    }
    virtual bool isShareable() const override { return true; }

    //! Finds material by name
    /** \return material index or -1 if not found
     **/
    int findMaterialByName(QString mtlName) const;

    //! Finds material by unique identifier
    /** \return material index or -1 if not found
     **/
    int findMaterialByUniqueID(QString uniqueID) const;

    //! Adds a material
    /** Ensures unicity of material names.
            \param mat material
            \param allowDuplicateNames whether to allow duplicate names for
    materials or not (in which case the returned index is the one of the
    material with the same name) \return material index
    **/
    int addMaterial(ccMaterial::CShared mat, bool allowDuplicateNames = false);

    //! MTL (material) file parser
    /** Inspired from KIXOR.NET "objloader"
     *(http://www.kixor.net/dev/objloader/)
     **/
    static bool ParseMTL(QString path,
                         const QString& filename,
                         ccMaterialSet& materials,
                         QStringList& errors);

    //! Saves to an MTL file (+ associated texture images)
    bool saveAsMTL(QString path,
                   const QString& baseFilename,
                   QStringList& errors) const;

    //! Clones materials set
    ccMaterialSet* clone() const;

    //! Appends materials from another set
    bool append(const ccMaterialSet& source);

    // inherited from ccSerializableObject
    virtual bool isSerializable() const override { return true; }

protected:
    // inherited from ccHObject
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;
};
