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
//#          COPYRIGHT: EDF R&D / TELECOM ParisTech (ENST-TSI)             #
//#                                                                        #
//##########################################################################

// Local
#include "ecvMaterial.h"

// CV_CORE_LIB
#include <CVTools.h>

// Qt
#include <QMap>
#include <QOpenGLContext>
#include <QOpenGLTexture>
#include <QUuid>

// Textures DB
static QMap<QString, QImage> s_textureDB;
// static QMap<QString, QSharedPointer<QOpenGLTexture>> s_openGLTextureDB;

ccMaterial::ccMaterial(const QString& name)
    : m_name(name),
      m_uniqueID(QUuid::createUuid().toString()),
      m_illum(1),
      m_diffuseFront(ecvColor::bright),
      m_diffuseBack(ecvColor::bright),
      m_ambient(ecvColor::night),
      m_specular(ecvColor::night),
      m_emission(ecvColor::night) {
    setShininess(50.0);
};

ccMaterial::ccMaterial(const ccMaterial& mtl)
    : m_name(mtl.m_name),
      m_textureFilename(mtl.m_textureFilename),
      m_uniqueID(mtl.m_uniqueID),
      m_illum(2),
      m_diffuseFront(mtl.m_diffuseFront),
      m_diffuseBack(mtl.m_diffuseBack),
      m_ambient(mtl.m_ambient),
      m_specular(mtl.m_specular),
      m_emission(mtl.m_emission),
      m_shininessFront(mtl.m_shininessFront),
      m_shininessBack(mtl.m_shininessFront) {}

void ccMaterial::setDiffuse(const ecvColor::Rgbaf& color) {
    setDiffuseFront(color);
    setDiffuseBack(color);
}

void ccMaterial::setShininess(float val) {
    setShininessFront(val);
    setShininessBack(0.8f * val);
}

void ccMaterial::setTransparency(float val) {
    m_diffuseFront.a = val;
    m_diffuseBack.a = val;
    m_ambient.a = val;
    m_specular.a = val;
    m_emission.a = val;
}

void ccMaterial::applyGL(const QOpenGLContext* context,
                         bool lightEnabled,
                         bool skipDiffuse) const {
    Q_UNUSED(context);
    Q_UNUSED(lightEnabled);
    Q_UNUSED(skipDiffuse);

    // get the set of OpenGL functions (version 2.1)
    // QOpenGLFunctions_2_1* glFunc =
    // context->versionFunctions<QOpenGLFunctions_2_1>(); assert(glFunc !=
    // nullptr);

    // if (glFunc == nullptr)
    //	return;

    // if (lightEnabled)
    //{
    //	if (!skipDiffuse)
    //	{
    //		glFunc->glMaterialfv(GL_FRONT, GL_DIFFUSE, m_diffuseFront.rgba);
    //		glFunc->glMaterialfv(GL_BACK,  GL_DIFFUSE, m_diffuseBack.rgba);
    //	}
    //	glFunc->glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   m_ambient.rgba);
    //	glFunc->glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  m_specular.rgba);
    //	glFunc->glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION,  m_emission.rgba);
    //	glFunc->glMaterialf (GL_FRONT,          GL_SHININESS, std::max(0.0f,
    // std::min(m_shininessFront, 128.0f))); 	glFunc->glMaterialf (GL_BACK,
    // GL_SHININESS, std::max(0.0f, std::min(m_shininessBack, 128.0f)));
    // }
    // else
    //{
    //	glFunc->glColor4fv(m_diffuseFront.rgba);
    // }
}

bool ccMaterial::loadAndSetTexture(const QString& absoluteFilename) {
    if (absoluteFilename.isEmpty()) {
        CVLog::Warning(QString(
                "[ccMaterial::loadAndSetTexture] filename can't be empty!"));
        return false;
    }

    // fix path separator bug
    const QString& nativeFilename =
            CVTools::ToNativeSeparators(absoluteFilename);

    if (s_textureDB.contains(nativeFilename)) {
        // if the image is already in memory, we simply update the texture
        // filename for this amterial
        m_textureFilename = nativeFilename;
    } else {
        // otherwise, we try to load the corresponding file
        QImage image(nativeFilename);
        if (image.isNull()) {
            CVLog::Warning(QString("[ccMaterial::loadAndSetTexture] Failed to "
                                   "load image '%1'")
                                   .arg(nativeFilename));
            return false;
        } else {
            setTexture(image, nativeFilename, false);
        }
    }

    return true;
}

void ccMaterial::setTexture(const QImage& image,
                            const QString& absoluteFilename /*=QString()*/,
                            bool mirrorImage /*=false*/) {
    CVLog::PrintDebug(QString("[ccMaterial::setTexture] absoluteFilename = "
                              "'%1' / size = %2 x %3")
                              .arg(absoluteFilename)
                              .arg(image.width())
                              .arg(image.height()));

    // fix path separator bug
    QString nativeFilename = CVTools::ToNativeSeparators(absoluteFilename);
    if (nativeFilename.isEmpty()) {
        // if the user hasn't provided any filename, we generate a fake one
        nativeFilename = QString("tex_%1.jpg").arg(m_uniqueID);
        assert(!s_textureDB.contains(nativeFilename));
    } else {
        // if the texture has already been loaded
        if (s_textureDB.contains(nativeFilename)) {
            // check that the size is compatible at least
            if (s_textureDB[nativeFilename].size() != image.size()) {
                CVLog::Warning(QString("[ccMaterial] A texture with the same "
                                       "name (%1) "
                                       "but with a different size has already "
                                       "been loaded!")
                                       .arg(nativeFilename));
            }
            m_textureFilename = nativeFilename;
            return;
        }
    }

    m_textureFilename = absoluteFilename;

    // insert image into DB if necessary
    s_textureDB[m_textureFilename] = mirrorImage ? image.mirrored() : image;
}

const QImage& ccMaterial::getTexture() const {
    return s_textureDB[m_textureFilename];
}

GLuint ccMaterial::getTextureID() const {
    if (QOpenGLContext::currentContext()) {
        const QImage& image = getTexture();
        if (image.isNull()) {
            return 0;
        } else {
            return 1;
        }
    } else {
        return 0;
    }
}

bool ccMaterial::hasTexture() const {
    return !m_textureFilename.isEmpty() &&
           !s_textureDB[m_textureFilename].isNull();
}

QImage ccMaterial::GetTexture(const QString& absoluteFilename) {
    return s_textureDB[absoluteFilename];
}

void ccMaterial::AddTexture(const QImage& image,
                            const QString& absoluteFilename) {
    s_textureDB[absoluteFilename] = image;
}

void ccMaterial::ReleaseTextures() {
    if (!QOpenGLContext::currentContext()) {
        CVLog::Warning("[ccMaterial::ReleaseTextures] No valid OpenGL context");
        return;
    }
}

void ccMaterial::releaseTexture() {
    if (m_textureFilename.isEmpty()) {
        // nothing to do
        return;
    }

    assert(QOpenGLContext::currentContext());

    s_textureDB.remove(m_textureFilename);
    m_textureFilename.clear();
}

bool ccMaterial::toFile(QFile& out) const {
    QDataStream outStream(&out);

    // material name (dataVersion >= 20)
    outStream << m_name;
    // texture (dataVersion >= 20)
    outStream << m_textureFilename;
    // material colors (dataVersion >= 20)
    // we don't use QByteArray here as it has its own versions!
    if (out.write((const char*)m_diffuseFront.rgba, sizeof(float) * 4) < 0)
        return WriteError();
    if (out.write((const char*)m_diffuseBack.rgba, sizeof(float) * 4) < 0)
        return WriteError();
    if (out.write((const char*)m_ambient.rgba, sizeof(float) * 4) < 0)
        return WriteError();
    if (out.write((const char*)m_specular.rgba, sizeof(float) * 4) < 0)
        return WriteError();
    if (out.write((const char*)m_emission.rgba, sizeof(float) * 4) < 0)
        return WriteError();
    // material shininess (dataVersion >= 20)
    outStream << m_shininessFront;
    outStream << m_shininessBack;

    return true;
}

bool ccMaterial::fromFile(QFile& in,
                          short dataVersion,
                          int flags,
                          LoadedIDMap& oldToNewIDMap) {
    Q_UNUSED(flags);

    QDataStream inStream(&in);

    // material name (dataVersion>=20)
    inStream >> m_name;
    if (dataVersion < 37) {
        // texture (dataVersion>=20)
        QImage texture;
        inStream >> texture;
        setTexture(texture, QString(), false);
    } else {
        // texture 'filename' (dataVersion>=37)
        inStream >> m_textureFilename;
    }
    // material colors (dataVersion>=20)
    if (in.read((char*)m_diffuseFront.rgba, sizeof(float) * 4) < 0)
        return ReadError();
    if (in.read((char*)m_diffuseBack.rgba, sizeof(float) * 4) < 0)
        return ReadError();
    if (in.read((char*)m_ambient.rgba, sizeof(float) * 4) < 0)
        return ReadError();
    if (in.read((char*)m_specular.rgba, sizeof(float) * 4) < 0)
        return ReadError();
    if (in.read((char*)m_emission.rgba, sizeof(float) * 4) < 0)
        return ReadError();
    // material shininess (dataVersion>=20)
    inStream >> m_shininessFront;
    inStream >> m_shininessBack;

    return true;
}

bool ccMaterial::compare(const ccMaterial& mtl) const {
    if (mtl.m_name != m_name || mtl.m_textureFilename != m_textureFilename ||
        mtl.m_shininessFront != m_shininessFront ||
        mtl.m_shininessBack != m_shininessBack || mtl.m_ambient != m_ambient ||
        mtl.m_specular != m_specular || mtl.m_emission != m_emission ||
        mtl.m_illum != m_illum || mtl.m_diffuseBack != m_diffuseBack ||
        mtl.m_diffuseFront != m_diffuseFront) {
        return false;
    }

    return true;
}
