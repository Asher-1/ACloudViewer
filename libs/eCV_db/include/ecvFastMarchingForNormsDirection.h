// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// cloudViewer
#include <DgmOctree.h>
#include <FastMarching.h>

// ECV_DB_LIB
#include "ecvAdvancedTypes.h"

// system
#include <vector>

class ccGenericPointCloud;
class ccPointCloud;
class ccOctree;
class ecvProgressDialog;

//! Fast Marching algorithm for normals direction resolution
/** Extends the FastMarching class.
 **/
class ccFastMarchingForNormsDirection : public cloudViewer::FastMarching {
public:
    //! Static entry point (helper)
    static int OrientNormals(ccPointCloud* theCloud,
                             unsigned char octreeLevel,
                             ecvProgressDialog* progressCb = nullptr);
    //! Default constructor
    ccFastMarchingForNormsDirection();

    //! Initializes the grid with a point cloud (and ist corresponding octree)
    /** The points should be associated to an (active) scalar field.
            The Fast Marching grid will have the same dimensions as
            the input octree considered at a given level of subdivision.
            \param cloud the point cloud
            \param theNorms the normals array
            \param theOctree the associated octree
            \param gridLevel the level of subdivision
            \return a negative value if something went wrong
    **/
    int init(ccGenericPointCloud* cloud,
             NormsIndexesTableType* theNorms,
             ccOctree* theOctree,
             unsigned char gridLevel);

    //! Updates a list of point flags, indicating the points already processed
    /** \return the number of resolved points
     **/
    unsigned updateResolvedTable(ccGenericPointCloud* theCloud,
                                 std::vector<unsigned char>& resolved,
                                 NormsIndexesTableType* theNorms);

    // inherited methods (see FastMarchingAlgorithm)
    int propagate() override;

protected:
    //! A Fast Marching grid cell for normals direction resolution
    class DirectionCell : public cloudViewer::FastMarching::Cell {
    public:
        //! Default constructor
        DirectionCell()
            : Cell(),
              N(0, 0, 0),
              C(0, 0, 0),
              cellCode(0),
              signConfidence(1)
#ifdef QT_DEBUG
              ,
              scalar(0)
#endif
        {
        }

        ///! Destructor
        ~DirectionCell() override = default;

        //! The local cell normal
        CCVector3 N;
        //! The local cell center
        CCVector3 C;
        //! the code of the equivalent cell in the octree
        cloudViewer::DgmOctree::CellCode cellCode;
        //! Confidence value
        float signConfidence;
#ifdef QT_DEBUG
        //! Undefined scalar for debug purposes
        float scalar;
#endif
    };

    // inherited methods (see FastMarchingAlgorithm)
    float computeTCoefApprox(
            cloudViewer::FastMarching::Cell* currentCell,
            cloudViewer::FastMarching::Cell* neighbourCell) const override;
    int step() override;
    void initTrialCells() override;
    bool instantiateGrid(unsigned size) override {
        return instantiateGridTpl<DirectionCell*>(size);
    }

    //! Computes relative 'confidence' between two cells (orientations)
    /** \return confidence between 0 and 1
     **/
    float computePropagationConfidence(DirectionCell* originCell,
                                       DirectionCell* destCell) const;

    //! Resolves the direction of a given cell (once and for all)
    void resolveCellOrientation(unsigned index);
};
