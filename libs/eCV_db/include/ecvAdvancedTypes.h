// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvArray.h"
#include "ecvBasicTypes.h"
#include "ecvColorTypes.h"
#include "ecvNormalCompressor.h"

/***************************************************
          Advanced CLOUDVIEWER  types (containers)
***************************************************/

//! Array of compressed 3D normals (single index)
class NormsIndexesTableType
    : public ccArray<CompressedNormType, 1, CompressedNormType> {
public:
    //! Default constructor
    ECV_DB_LIB_API NormsIndexesTableType();
    ~NormsIndexesTableType() override = default;

    // inherited from ccArray/ccHObject
    CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::NORMAL_INDEXES_ARRAY;
    }

    //! Duplicates array (overloaded from ccArray::clone)
    NormsIndexesTableType* clone() override {
        NormsIndexesTableType* cloneArray = new NormsIndexesTableType();
        if (!copy(*cloneArray)) {
            CVLog::Warning(
                    "[NormsIndexesTableType::clone] Failed to clone array (not "
                    "enough memory)");
            cloneArray->release();
            return nullptr;
        }
        cloneArray->setName(getName());
        return cloneArray;
    }

    // inherited from ccHObject/ccArray
    ECV_DB_LIB_API bool fromFile_MeOnly(QFile& in,
                                        short dataVersion,
                                        int flags,
                                        LoadedIDMap& oldToNewIDMap) override;
};

//! Array of (uncompressed) 3D normals (Nx,Ny,Nz)
class NormsTableType : public ccArray<CCVector3, 3, PointCoordinateType> {
public:
    //! Default constructor
    NormsTableType() : ccArray<CCVector3, 3, PointCoordinateType>("Normals") {}
    virtual ~NormsTableType() = default;

    // inherited from ccArray/ccHObject
    CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::NORMALS_ARRAY;
    }

    //! Duplicates array (overloaded from ccArray::clone)
    NormsTableType* clone() override {
        NormsTableType* cloneArray = new NormsTableType();
        if (!copy(*cloneArray)) {
            CVLog::Warning(
                    "[NormsTableType::clone] Failed to clone array (not enough "
                    "memory)");
            cloneArray->release();
            return nullptr;
        }
        cloneArray->setName(getName());
        return cloneArray;
    }
};

//! Array of RGB colors for each point
class ColorsTableType : public ccArray<ecvColor::Rgb, 3, ColorCompType> {
public:
    //! Default constructor
    ColorsTableType()
        : ccArray<ecvColor::Rgb, 3, ColorCompType>("RGB colors") {}
    virtual ~ColorsTableType() = default;

    // inherited from ccArray/ccHObject
    CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::RGB_COLOR_ARRAY;
    }

    //! Duplicates array (overloaded from ccArray::clone)
    ColorsTableType* clone() override {
        ColorsTableType* cloneArray = new ColorsTableType();
        if (!copy(*cloneArray)) {
            CVLog::Warning(
                    "[ColorsTableType::clone] Failed to clone array (not "
                    "enough memory)");
            cloneArray->release();
            return nullptr;
        }
        cloneArray->setName(getName());
        return cloneArray;
    }
};

//! Array of RGBA colors for each point
class RGBAColorsTableType : public ccArray<ecvColor::Rgba, 4, ColorCompType> {
public:
    //! Default constructor
    RGBAColorsTableType()
        : ccArray<ecvColor::Rgba, 4, ColorCompType>("RGBA colors") {}
    virtual ~RGBAColorsTableType() = default;

    // inherited from ccArray/ccHObject
    CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::RGBA_COLOR_ARRAY;
    }

    //! Duplicates array (overloaded from ccArray::clone)
    RGBAColorsTableType* clone() override {
        RGBAColorsTableType* cloneArray = new RGBAColorsTableType();
        if (!copy(*cloneArray)) {
            CVLog::Warning(
                    "[RGBAColorsTableType::clone] Failed to clone array (not "
                    "enough memory)");
            cloneArray->release();
            return nullptr;
        }
        cloneArray->setName(getName());
        return cloneArray;
    }
};

//! 2D texture coordinates
struct TexCoords2D {
    TexCoords2D() : tx(-1.0f), ty(-1.0f) {}
    TexCoords2D(float x, float y) : tx(x), ty(y) {}

    union {
        struct {
            float tx, ty;
        };
        float t[2];
    };
};

//! Array of 2D texture coordinates
class TextureCoordsContainer : public ccArray<TexCoords2D, 2, float> {
public:
    //! Default constructor
    TextureCoordsContainer()
        : ccArray<TexCoords2D, 2, float>("Texture coordinates") {}
    virtual ~TextureCoordsContainer() = default;

    // inherited from ccArray/ccHObject
    CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::TEX_COORDS_ARRAY;
    }

    //! Duplicates array (overloaded from ccArray::clone)
    TextureCoordsContainer* clone() override {
        TextureCoordsContainer* cloneArray = new TextureCoordsContainer();
        if (!copy(*cloneArray)) {
            CVLog::Warning(
                    "[TextureCoordsContainer::clone] Failed to clone array "
                    "(not enough memory)");
            cloneArray->release();
            return nullptr;
        }
        cloneArray->setName(getName());
        return cloneArray;
    }
};
