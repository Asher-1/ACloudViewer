// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_GENERIC_VISUALIZER_2D_HEADER
#define ECV_GENERIC_VISUALIZER_2D_HEADER

#include "ecvGenericVisualizer.h"

//! Generic visualizer 2D interface
class ECV_DB_LIB_API ecvGenericVisualizer2D : public ecvGenericVisualizer {
    Q_OBJECT

public:
    //! Default constructor
    /** \param name object name
     **/
    ecvGenericVisualizer2D() = default;

    //! Destructor
    virtual ~ecvGenericVisualizer2D() = default;
};

#endif  // ECV_GENERIC_VISUALIZER_2D_HEADER
