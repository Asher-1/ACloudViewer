// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_IMAGE_HEADER
#define ECV_IMAGE_HEADER

// Local
#include "ecvHObject.h"

// Qt
#include <QImage>

class ccCameraSensor;

//! Generic image
class ECV_DB_LIB_API ccImage : public ccHObject {
public:
    //! Default constructor
    ccImage();

    //! Constructor from QImage
    ccImage(const QImage& image, const QString& name = QString("unknown"));

    // inherited methods (ccHObject)
    virtual bool isSerializable() const override { return true; }

    //! Returns unique class ID
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::IMAGE;
    }

    //! Loads image from file
    /** \param filename image filename
            \param error a human readable description of what went wrong (if
    method fails) \return success
    **/
    bool load(const QString& filename, QString& error);

    //! Returns image data
    inline QImage& data() { return m_image; }
    //! Returns image data (const version)
    inline const QImage& data() const { return m_image; }

    //! Sets image data
    void setData(const QImage& image);

    //! Returns image width
    inline unsigned getW() const { return m_width; }

    //! Returns image height
    inline unsigned getH() const { return m_height; }

    //! Sets image texture transparency
    void setAlpha(float value);

    //! Returns image texture transparency
    inline float getAlpha() const { return m_texAlpha; }

    //! Manually sets aspect ratio
    void setAspectRatio(float ar) { m_aspectRatio = ar; }

    //! Returns aspect ratio
    inline float getAspectRatio() const { return m_aspectRatio; }

    //! Sets associated sensor
    void setAssociatedSensor(ccCameraSensor* sensor);

    //! Returns associated sensor
    ccCameraSensor* getAssociatedSensor() { return m_associatedSensor; }

    //! Returns associated sensor (const version)
    const ccCameraSensor* getAssociatedSensor() const {
        return m_associatedSensor;
    }

protected:
    // inherited from ccHObject
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;
    virtual void onDeletionOf(const ccHObject* obj) override;
    virtual bool toFile_MeOnly(QFile& out) const override;
    virtual bool fromFile_MeOnly(QFile& in,
                                 short dataVersion,
                                 int flags,
                                 LoadedIDMap& oldToNewIDMap) override;

    //! Updates aspect ratio
    void updateAspectRatio();

    //! Image width (in pixels)
    unsigned m_width;
    //! Image height (in pixels)
    unsigned m_height;

    //! Aspect ratio w/h
    /** Default is m_width/m_height.
            Should be changed if pixels are not square.
    **/
    float m_aspectRatio;

    //! Texture transparency
    float m_texAlpha;

    //! Image data
    QImage m_image;

    //! Associated sensor
    ccCameraSensor* m_associatedSensor;
};

#endif  // CC_IMAGE_HEADER
