// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "CV_db.h"

// QT
#include <QObject>

//! Generic visualizer interface
class CV_DB_LIB_API ecvGenericVisualizer : public QObject {
    Q_OBJECT

public:
    //! Default constructor
    /** \param name object name
     **/
    ecvGenericVisualizer() = default;

    //! Destructor
    virtual ~ecvGenericVisualizer() = default;
};
