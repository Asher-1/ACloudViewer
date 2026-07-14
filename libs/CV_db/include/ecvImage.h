// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvHObject.h"

// Qt
#include <QImage>

#include <memory>
#include <vector>

class ccCameraSensor;

//! Generic image
class CV_DB_LIB_API ccImage : public ccHObject {
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

    // --- DA3 depth estimation interface ---
    // These methods are declared unconditionally (not guarded by #ifdef DA3_ENABLED)
    // because CV_DB_LIB links DA3Core with PRIVATE visibility — consumers of
    // CV_DB_LIB do not inherit the DA3_ENABLED definition. The implementations
    // gracefully return false when DA3 is disabled; use isDA3Available() for
    // runtime feature detection.

    //! Depth estimation result for a single image
    struct DepthResult {
        std::vector<float> depth;
        std::vector<float> confidence;
        int width = 0;
        int height = 0;
        bool has_pose = false;
        float extrinsics[12] = {};
        float intrinsics[9] = {};
    };

    //! Estimates monocular depth from the loaded image using DA3.
    //! Returns true on success; result is stored in out.
    //! \param model_path path to GGUF model file
    //! \param n_threads number of inference threads (0 = default)
    //! \param out depth estimation output
    //! \param metric_model_path optional path to nested metric model
    bool estimateDepth(const QString& model_path, int n_threads,
                       DepthResult& out,
                       const QString& metric_model_path = QString()) const;

    //! Estimates depth + camera pose from the loaded image using DA3.
    //! \param model_path path to GGUF model file
    //! \param n_threads number of inference threads (0 = default)
    //! \param out depth + pose estimation output (has_pose=true on success)
    //! \param metric_model_path optional path to nested metric model
    bool estimateDepthAndPose(const QString& model_path, int n_threads,
                              DepthResult& out,
                              const QString& metric_model_path = QString()) const;

    //! Returns true if DA3 depth estimation is available (compiled with DA3_ENABLED)
    static bool isDA3Available();

protected:
    // inherited from ccHObject
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;
    virtual void onDeletionOf(const ccHObject* obj) override;
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
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
