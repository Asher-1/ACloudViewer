// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvOctreeSpinBox.h"

// cloudViewer
#include <CVMiscTools.h>

// CV_DB_LIB
#include "ecvBBox.h"
#include "ecvGenericPointCloud.h"
#include "ecvOctree.h"

ccOctreeSpinBox::ccOctreeSpinBox(QWidget* parent /*=0*/)
    : QSpinBox(parent), m_octreeBoxWidth(0) {
    setRange(0, cloudViewer::DgmOctree::MAX_OCTREE_LEVEL);

    // we'll catch any modification of the spinbox value and update the suffix
    // consequently
    connect(this, SIGNAL(valueChanged(int)), this, SLOT(onValueChange(int)));
}

void ccOctreeSpinBox::setCloud(ccGenericPointCloud* cloud) {
    if (!cloud) {
        assert(false);
        return;
    }

    if (cloud->getOctree()) {
        setOctree(cloud->getOctree().data());
    } else {
        ccBBox box = cloud->getOwnBB(false);
        cloudViewer::CCMiscTools::MakeMinAndMaxCubical(box.minCorner(),
                                                       box.maxCorner());
        m_octreeBoxWidth = box.getMaxBoxDim();
        onValueChange(value());
    }
}

void ccOctreeSpinBox::setOctree(cloudViewer::DgmOctree* octree) {
    if (octree) {
        m_octreeBoxWidth = static_cast<double>(octree->getCellSize(0));
        onValueChange(value());
    } else {
        m_octreeBoxWidth = 0;
        setSuffix(QString());
    }
}

void ccOctreeSpinBox::onValueChange(int level) {
    if (m_octreeBoxWidth > 0) {
        if (level >=
            0 /* && level <= cloudViewer::DgmOctree::MAX_OCTREE_LEVEL*/) {
            double cs = m_octreeBoxWidth / pow(2.0, static_cast<double>(level));
            setSuffix(QString(" (grid step = %1)").arg(cs));
        } else {
            // invalid level?!
            setSuffix(QString());
        }
    }
}
