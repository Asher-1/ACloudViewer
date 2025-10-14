// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_2D_VIEWPORT_OBJECT_HEADER
#define ECV_2D_VIEWPORT_OBJECT_HEADER

// Local
#include "ecvHObject.h"
#include "ecvViewportParameters.h"

//! 2D viewport object
class ECV_DB_LIB_API cc2DViewportObject : public ccHObject {
public:
    //! Default constructor
    cc2DViewportObject(QString name = QString());

    // inherited from ccHObject
    virtual CV_CLASS_ENUM getClassID() const override {
        return CV_TYPES::VIEWPORT_2D_OBJECT;
    }
    virtual bool isSerializable() const override { return true; }

    //! Sets perspective view state
    void setParameters(const ecvViewportParameters& params) {
        m_params = params;
    }

    //! Gets parameters
    const ecvViewportParameters& getParameters() const { return m_params; }

protected:
    // inherited from ccHObject
    virtual bool toFile_MeOnly(QFile& out) const override;
    virtual bool fromFile_MeOnly(QFile& in,
                                 short dataVersion,
                                 int flags,
                                 LoadedIDMap& oldToNewIDMap) override;

    //! Viewport parameters
    ecvViewportParameters m_params;
};

#endif  // CC_2D_VIEWPORT_OBJECT_HEADER
