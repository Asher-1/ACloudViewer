// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVGeom.h>

#include "ecvGenericVisualizer.h"

//! Generic visualizer 3D interface
class ECV_DB_LIB_API ecvGenericVisualizer3D : public ecvGenericVisualizer {
    Q_OBJECT

public:
    //! Default constructor
    /** \param name object name
     **/
    ecvGenericVisualizer3D() = default;

    //! Destructor
    virtual ~ecvGenericVisualizer3D() = default;

signals:
    void interactorKeyboardEvent(const std::string& symKey);
    void interactorPointPickedEvent(const CCVector3& p,
                                    int index,
                                    const std::string& id);
    void interactorAreaPickedEvent(const std::vector<int>& selected_slice);
};
