// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ReferenceCloud.h"

namespace cloudViewer {

//! A simple polyline class
/** The polyline is considered as a cloud of points
        (in a specific order) with a open/closed state
        information.
**/
class CV_CORE_LIB_API Polyline : public ReferenceCloud {
public:
    //! Polyline constructor
    explicit Polyline(GenericIndexedCloudPersist* associatedCloud);

    //! Returns whether the polyline is closed or not
    inline bool isClosed() const { return m_isClosed; }

    //! Sets whether the polyline is closed or not
    inline void setClosed(bool state) { m_isClosed = state; }

    // inherited from ReferenceCloud
    void clear(bool unusedParam = true) override;

protected:
    //! Closing state
    bool m_isClosed;
};

}  // namespace cloudViewer
