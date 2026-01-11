// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecv2DViewportObject.h"

//! 2D viewport label
class ECV_DB_LIB_API cc2DViewportLabel : public cc2DViewportObject {
public:
    //! Default constructor
    explicit cc2DViewportLabel(QString name = QString());

    // inherited from ccHObject
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::VIEWPORT_2D_LABEL;
    }
    virtual bool isSerializable() const override { return true; }

    //! Returns ROI (relative to screen)
    const float* roi() const { return m_roi; }

    //! Sets ROI (relative to screen)
    void setRoi(const float* roi);

    void clear2Dviews();

    void updateLabel();

    void update2DLabelView(CC_DRAW_CONTEXT& context, bool updateScreen = true);

protected:
    // inherited from ccHObject
    bool toFile_MeOnly(QFile& out, short dataVersion) const override;
    short minimumFileVersion_MeOnly() const override;
    bool fromFile_MeOnly(QFile& in,
                         short dataVersion,
                         int flags,
                         LoadedIDMap& oldToNewIDMap) override;

    //! Draws the entity only (not its children)
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

    //! label ROI
    /** ROI is relative to screen
     **/
    float m_roi[4];
};
