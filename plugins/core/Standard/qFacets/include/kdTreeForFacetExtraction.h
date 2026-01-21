// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CV_DB_LIB
#include <ecvKdTree.h>

class ccKdTreeForFacetExtraction {
public:
    //! Default constructor
    ccKdTreeForFacetExtraction();

    //! Fuses cells
    /** Creates a new scalar fields with the groups indexes.
            \param kdTree Kd-tree
            \param maxError max error after fusion (see errorMeasure)
            \param errorMeasure error measure type
            \param maxAngle_deg maximum angle between two sets to allow fusion
    (in degrees) \param overlapCoef maximum relative distance between two sets
    to accept fusion (1 = no distance, < 1 = overlap, > 1 = gap) \param
    closestFirst \param progressCb for progress notifications (optional)
    **/
    static bool FuseCells(
            ccKdTree* kdTree,
            double maxError,
            cloudViewer::DistanceComputationTools::ERROR_MEASURES errorMeasure,
            double maxAngle_deg,
            PointCoordinateType overlapCoef = 1,
            bool closestFirst = true,
            cloudViewer::GenericProgressCallback* progressCb = 0);
};
