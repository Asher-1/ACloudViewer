// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "CV_db.h"

// cloudViewer
#include <DgmOctree.h>

// Qt
#include <QSpinBox>

class ccGenericPointCloud;

//! Octree level editor dialog
class CV_DB_LIB_API ccOctreeSpinBox : public QSpinBox {
    Q_OBJECT

public:
    //! Default constructor
    explicit ccOctreeSpinBox(QWidget* parent = 0);

    //! Inits the dialog with a cloud (on which the octree has been or will be
    //! computed)
    /** Alternative to ccOctreeSpinBox::setOctree
     **/
    void setCloud(ccGenericPointCloud* cloud);

    //! Inits the dialog with an octree
    /** Alternative to ccOctreeSpinBox::setCloud
     **/
    void setOctree(cloudViewer::DgmOctree* octree);

protected slots:

    //! Called each time the spinbox value changes
    void onValueChange(int);

protected:
    //! Corresponding octree base size
    double m_octreeBoxWidth;
};
