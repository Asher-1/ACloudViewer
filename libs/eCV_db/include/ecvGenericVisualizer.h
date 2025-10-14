// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_GENERIC_VISUALIZER_HEADER
#define ECV_GENERIC_VISUALIZER_HEADER

// LOCAL
#include "eCV_db.h"

// QT
#include <QObject>

//! Generic visualizer interface
class ECV_DB_LIB_API ecvGenericVisualizer : public QObject {
    Q_OBJECT

public:
    //! Default constructor
    /** \param name object name
     **/
    ecvGenericVisualizer() = default;

    //! Destructor
    virtual ~ecvGenericVisualizer() = default;
};

#endif  // ECV_GENERIC_VISUALIZER_HEADER
