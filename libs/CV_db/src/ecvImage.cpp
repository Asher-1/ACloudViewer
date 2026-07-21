// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvImage.h"

// Local
#include "ecvCameraSensor.h"
#include "ecvGenericGLDisplay.h"

// Qt
#include <QFileInfo>
#include <QImageReader>
#include <QOpenGLTexture>
#include <algorithm>
#include <cstring>

#ifdef AICore_ENABLED
#include "aicore/depth_image.h"
#endif

ccImage::ccImage()
    : ccHObject("Not loaded"),
      m_width(0),
      m_height(0),
      m_aspectRatio(1.0f),
      m_texAlpha(1.0f),
      m_associatedSensor(0) {
    setVisible(true);
    lockVisibility(false);
    setEnabled(false);
}

ccImage::ccImage(const QImage& image, const QString& name)
    : ccHObject(name),
      m_width(image.width()),
      m_height(image.height()),
      m_aspectRatio(1.0f),
      m_texAlpha(1.0f),
      m_image(image),
      m_associatedSensor(0) {
    updateAspectRatio();
    setVisible(true);
    lockVisibility(false);
    setEnabled(true);
}

bool ccImage::load(const QString& filename, QString& error) {
    QImageReader reader(filename);
    // Automatically apply EXIF orientation metadata so that
    // images taken with a rotated camera are displayed upright.
    reader.setAutoTransform(true);
    QImage image = reader.read();
    if (image.isNull()) {
        error = reader.errorString();
        return false;
    }

    setData(image);

    setName(QFileInfo(filename).fileName());
    setEnabled(true);

    return true;
}

void ccImage::setData(const QImage& image) {
    m_image = image;
    m_width = m_image.width();
    m_height = m_image.height();
    updateAspectRatio();
    setRedraw(true);
}

void ccImage::updateAspectRatio() {
    setAspectRatio(m_height != 0 ? static_cast<float>(m_width) / m_height
                                 : 1.0f);
}

void ccImage::drawMeOnly(CC_DRAW_CONTEXT& context) {
    if (m_image.isNull()) return;

    if (!MACRO_Draw2D(context) || !MACRO_Foreground(context)) return;

    if (!context.display) return;

    context.display->draw(context, this);
}

void ccImage::setAlpha(float value) {
    const float clamped = value <= 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    if (m_texAlpha == clamped) {
        return;
    }
    m_texAlpha = clamped;
    setRedraw(true);
}

ccBBox ccImage::getOwnBB(bool withGLFeatures /*=false*/) {
    Q_UNUSED(withGLFeatures);
    if (m_width == 0 || m_height == 0) {
        return ccBBox();
    }
    ccBBox box;
    box.add(CCVector3(0, 0, 0));
    box.add(CCVector3(static_cast<PointCoordinateType>(m_width - 1),
                      static_cast<PointCoordinateType>(m_height - 1), 0));
    return box;
}

void ccImage::setAssociatedSensor(ccCameraSensor* sensor) {
    m_associatedSensor = sensor;

    if (m_associatedSensor)
        m_associatedSensor->addDependency(this, DP_NOTIFY_OTHER_ON_DELETE);
}

void ccImage::onDeletionOf(const ccHObject* obj) {
    if (obj == m_associatedSensor) setAssociatedSensor(0);

    ccHObject::onDeletionOf(obj);
}

bool ccImage::toFile_MeOnly(QFile& out, short dataVersion) const {
    assert(out.isOpen() && (out.openMode() & QIODevice::WriteOnly));
    if (dataVersion < 38) {
        assert(false);
        return false;
    }

    if (!ccHObject::toFile_MeOnly(out, dataVersion)) return false;

    // we can't save the associated sensor here (as it may be shared by multiple
    // images) so instead we save it's unique ID (dataVersion>=38) WARNING: the
    // sensor must be saved in the same BIN file! (responsibility of the caller)
    uint32_t sensorUniqueID =
            (m_associatedSensor ? (uint32_t)m_associatedSensor->getUniqueID()
                                : 0);
    if (out.write((const char*)&sensorUniqueID, 4) < 0) return WriteError();

    // for backward compatibility
    float texU = 1.0f, texV = 1.0f;

    QDataStream outStream(&out);
    outStream << m_width;
    outStream << m_height;
    outStream << m_aspectRatio;
    outStream << texU;
    outStream << texV;
    outStream << m_texAlpha;
    outStream << m_image;
    QString fakeString;
    outStream << fakeString;  // formerly: 'complete filename'

    return true;
}

short ccImage::minimumFileVersion_MeOnly() const {
    return std::max(static_cast<short>(38),
                    ccHObject::minimumFileVersion_MeOnly());
}

bool ccImage::fromFile_MeOnly(QFile& in,
                              short dataVersion,
                              int flags,
                              LoadedIDMap& oldToNewIDMap) {
    if (!ccHObject::fromFile_MeOnly(in, dataVersion, flags, oldToNewIDMap))
        return false;

    // as the associated sensor can't be saved directly (as it may be shared by
    // multiple images) we only store its unique ID (dataVersion >= 38) --> we
    // hope we will find it at loading time (i.e. this is the responsibility of
    // the caller to make sure that all dependencies are saved together)
    uint32_t sensorUniqueID = 0;
    if (in.read((char*)&sensorUniqueID, 4) < 0) return ReadError();
    //[DIRTY] WARNING: temporarily, we set the vertices unique ID in the
    //'m_associatedCloud' pointer!!!
    *(uint32_t*)(&m_associatedSensor) = sensorUniqueID;

    float texU, texV;

    QDataStream inStream(&in);
    inStream >> m_width;
    inStream >> m_height;
    inStream >> m_aspectRatio;
    inStream >> texU;
    inStream >> texV;
    inStream >> m_texAlpha;
    inStream >> m_image;
    QString fakeString;
    inStream >> fakeString;  // formerly: 'complete filename'

    return true;
}

#ifdef AICore_ENABLED
namespace {

void copyDepthResult(const aicore::depth::ImageDepthResult& src,
                     ccImage::DepthResult& out) {
    out.width = src.width;
    out.height = src.height;
    out.depth = src.depth;
    out.confidence = src.confidence;
    out.has_pose = src.has_pose;
    std::memcpy(out.extrinsics, src.extrinsics, sizeof(out.extrinsics));
    std::memcpy(out.intrinsics, src.intrinsics, sizeof(out.intrinsics));
}

}  // namespace
#endif

bool ccImage::estimateDepth(const QString& model_path,
                            int n_threads,
                            DepthResult& out,
                            const QString& metric_model_path) const {
#ifndef AICore_ENABLED
    Q_UNUSED(model_path);
    Q_UNUSED(n_threads);
    Q_UNUSED(out);
    Q_UNUSED(metric_model_path);
    return false;
#else
    if (m_image.isNull()) return false;

    aicore::depth::ImageDepthResult result;
    if (!aicore::depth::ImageDepth::estimateDepth(
                m_image, model_path, n_threads, result, metric_model_path)) {
        return false;
    }
    copyDepthResult(result, out);
    return true;
#endif
}

bool ccImage::estimateDepthAndPose(const QString& model_path,
                                   int n_threads,
                                   DepthResult& out,
                                   const QString& metric_model_path) const {
#ifndef AICore_ENABLED
    Q_UNUSED(model_path);
    Q_UNUSED(n_threads);
    Q_UNUSED(out);
    Q_UNUSED(metric_model_path);
    return false;
#else
    if (m_image.isNull()) return false;

    aicore::depth::ImageDepthResult result;
    if (!aicore::depth::ImageDepth::estimateDepthAndPose(
                m_image, model_path, n_threads, result, metric_model_path)) {
        return false;
    }
    copyDepthResult(result, out);
    return true;
#endif
}

bool ccImage::isAICoreAvailable() {
#ifdef AICore_ENABLED
    return aicore::depth::ImageDepth::isAvailable();
#else
    return false;
#endif
}
