// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvColorTypes.h"
#include "ecvSerializableObject.h"

// Qt
#include <QtGui/qopengl.h>

#include <QSharedPointer>

// STL
#include <map>
#include <utility>
#include <vector>

class QImage;
class QOpenGLContext;

//! Mesh (triangle) material
class ECV_DB_LIB_API ccMaterial : public ccSerializableObject {
public:
    //! Const + Shared type
    typedef QSharedPointer<const ccMaterial> CShared;
    //! Shared type
    typedef QSharedPointer<ccMaterial> Shared;

    //! Default constructor
    explicit ccMaterial(const QString& name = QString("default"));

    //! Copy constructor
    ccMaterial(const ccMaterial& mtl);

    //! Returns the material name
    inline const QString& getName() const { return m_name; }
    //! Returns the texture filename (if any)
    inline const QString& getTextureFilename() const {
        return m_textureFilename;
    }
    //! Sets the material name
    inline void setName(QString name) { m_name = name; }

    //! Sets diffuse color (both front and back)
    void setDiffuse(const ecvColor::Rgbaf& color);
    //! Sets diffuse color (front)
    inline void setDiffuseFront(const ecvColor::Rgbaf& color) {
        m_diffuseFront = color;
    }
    //! Sets diffuse color (back)
    inline void setDiffuseBack(const ecvColor::Rgbaf& color) {
        m_diffuseBack = color;
    }
    //! Returns front diffuse color
    inline const ecvColor::Rgbaf& getDiffuseFront() const {
        return m_diffuseFront;
    }
    //! Returns back diffuse color
    inline const ecvColor::Rgbaf& getDiffuseBack() const {
        return m_diffuseBack;
    }

    //! Sets ambient color
    inline void setAmbient(const ecvColor::Rgbaf& color) { m_ambient = color; }
    //! Returns ambient color
    inline const ecvColor::Rgbaf& getAmbient() const { return m_ambient; }

    //! Sets illum mode
    inline void setIllum(int illum) { m_illum = illum; }
    //! Returns illum mode
    inline int getIllum() const { return m_illum; }

    //! Sets specular color
    inline void setSpecular(const ecvColor::Rgbaf& color) {
        m_specular = color;
    }
    //! Returns specular color
    inline const ecvColor::Rgbaf& getSpecular() const { return m_specular; }

    //! Sets emission color
    inline void setEmission(const ecvColor::Rgbaf& color) {
        m_emission = color;
    }
    //! Returns emission color
    inline const ecvColor::Rgbaf& getEmission() const { return m_emission; }

    //! Sets shininess (both front - 100% - and back - 80%)
    void setShininess(float val);
    //! Sets shininess (front)
    inline void setShininessFront(float val) { m_shininessFront = val; }
    //! Sets shininess (back)
    inline void setShininessBack(float val) { m_shininessBack = val; }
    //! Returns front shininess
    inline float getShininessFront() const { return m_shininessFront; }
    //! Returns back shininess
    inline float getShininessBack() const { return m_shininessBack; }

    //! Sets transparency (all colors)
    void setTransparency(float val);

    //! Sets metallic factor (PBR)
    inline void setMetallic(float val) { m_metallic = val; }
    //! Returns metallic factor
    inline float getMetallic() const { return m_metallic; }

    //! Sets roughness factor (PBR)
    inline void setRoughness(float val) { m_roughness = val; }
    //! Returns roughness factor
    inline float getRoughness() const { return m_roughness; }

    //! Sets sheen factor (PBR)
    inline void setSheen(float val) { m_sheen = val; }
    //! Returns sheen factor
    inline float getSheen() const { return m_sheen; }

    //! Sets clearcoat factor (PBR)
    inline void setClearcoat(float val) { m_clearcoat = val; }
    //! Returns clearcoat factor
    inline float getClearcoat() const { return m_clearcoat; }

    //! Sets clearcoat roughness (PBR)
    inline void setClearcoatRoughness(float val) { m_clearcoatRoughness = val; }
    //! Returns clearcoat roughness
    inline float getClearcoatRoughness() const { return m_clearcoatRoughness; }

    //! Sets anisotropy factor (PBR)
    inline void setAnisotropy(float val) { m_anisotropy = val; }
    //! Returns anisotropy factor
    inline float getAnisotropy() const { return m_anisotropy; }

    //! Sets ambient occlusion factor (PBR)
    inline void setAmbientOcclusion(float val) { m_ambientOcclusion = val; }
    //! Returns ambient occlusion factor
    inline float getAmbientOcclusion() const { return m_ambientOcclusion; }

    //! Apply parameters (OpenGL)
    void applyGL(const QOpenGLContext* context,
                 bool lightEnabled,
                 bool skipDiffuse) const;

    //! Returns whether the material has an associated texture or not
    bool hasTexture() const;

    //! Sets texture
    /** If no filename is provided, a random one will be generated.
     **/
    void setTexture(const QImage& image,
                    const QString& absoluteFilename = QString(),
                    bool mirrorImage = false);

    //! Loads texture from file (and set it if successful)
    /** If the filename is not already in DB, the corresponding file will be
    loaded. \return whether the file could be loaded (or is already in DB) or
    not
    **/
    bool loadAndSetTexture(const QString& absoluteFilename);

    //! Returns the texture (if any)
    const QImage& getTexture() const;

    //! Returns the texture ID (if any)
    GLuint getTextureID() const;

    //! Returns the texture image associated to a given name
    static QImage GetTexture(const QString& absoluteFilename);

    // ========== Multi-Texture PBR Support ==========

    //! Texture map types for PBR materials
    enum class TextureMapType {
        DIFFUSE,       // map_Kd - Diffuse/Albedo
        AMBIENT,       // map_Ka - Ambient Occlusion
        SPECULAR,      // map_Ks - Specular
        NORMAL,        // map_Bump, map_bump, norm, bump - Normal map
        METALLIC,      // map_Pm - Metallic
        ROUGHNESS,     // map_Pr - Roughness
        SHININESS,     // map_Ns - Shininess/Glossiness (inverse of roughness)
        EMISSIVE,      // map_Ke - Emissive
        OPACITY,       // map_d - Opacity/Alpha
        DISPLACEMENT,  // map_disp, disp - Displacement
        REFLECTION,    // refl - Reflection
        SHEEN,         // map_Ps - Sheen (fabric-like materials)
        CLEARCOAT,     // map_Pc - Clearcoat layer
        CLEARCOAT_ROUGHNESS,  // map_Pcr - Clearcoat roughness
        ANISOTROPY            // map_aniso - Anisotropic reflection
    };

    //! Load and set a specific texture map type
    bool loadAndSetTextureMap(TextureMapType type,
                              const QString& absoluteFilename);

    //! Get texture filename for a specific map type
    QString getTextureFilename(TextureMapType type) const;

    //! Check if a specific texture map type exists
    bool hasTextureMap(TextureMapType type) const;

    //! Get all texture map filenames
    std::vector<std::pair<TextureMapType, QString>> getAllTextureFilenames()
            const;

    //! Adds a texture to the global texture DB
    static void AddTexture(const QImage& image,
                           const QString& absoluteFilename);

    //! Release all texture objects
    /** Should be called BEFORE the global shared context is destroyed.
     **/
    static void ReleaseTextures();

    //! Release the texture
    /** \warning Make sure no more materials are using this texture!
     **/
    void releaseTexture();

    //! Compares this material with another one
    /** \return true if both materials are equivalent or false otherwise
     **/
    bool compare(const ccMaterial& mtl) const;

    // inherited from ccSerializableObject
    bool isSerializable() const override { return true; }
    /** \warning Doesn't save the texture image!
     **/
    bool toFile(QFile& out) const override;
    virtual bool fromFile(QFile& in,
                          short dataVersion,
                          int flags,
                          LoadedIDMap& oldToNewIDMap);

    //! Returns unique identifier (UUID)
    inline QString getUniqueIdentifier() const { return m_uniqueID; }

protected:
    QString m_name;
    QString m_textureFilename;  // Legacy: main diffuse texture
    QString m_uniqueID;
    int m_illum;

    ecvColor::Rgbaf m_diffuseFront;
    ecvColor::Rgbaf m_diffuseBack;
    ecvColor::Rgbaf m_ambient;
    ecvColor::Rgbaf m_specular;
    ecvColor::Rgbaf m_emission;
    float m_shininessFront;
    float m_shininessBack;

    // PBR scalar parameters
    float m_metallic;            // Pm - Metallic factor [0,1]
    float m_roughness;           // Pr - Roughness factor [0,1]
    float m_sheen;               // Ps - Sheen factor [0,1]
    float m_clearcoat;           // Pc - Clearcoat factor [0,1]
    float m_clearcoatRoughness;  // Pcr - Clearcoat roughness [0,1]
    float m_anisotropy;          // aniso - Anisotropy factor [0,1]
    float m_ambientOcclusion;    // Pa - Ambient Occlusion factor [0,1]

    // Multi-texture PBR support
    std::map<TextureMapType, QString> m_textureFilenames;
};
