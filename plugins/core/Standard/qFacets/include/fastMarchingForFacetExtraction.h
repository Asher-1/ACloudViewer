// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// cloudViewer
#include <DistanceComputationTools.h>
#include <FastMarching.h>
#include <GenericProgressCallback.h>

// ECV_DB_LIB
#include <ecvAdvancedTypes.h>

class ccGenericPointCloud;
class ccPointCloud;

//! Fast Marching algorithm for planar facets extraction (qFacets plugin)
/** Extends the FastMarching class.
 **/
class FastMarchingForFacetExtraction : public cloudViewer::FastMarching {
public:
    //! Static entry point (helper)
    static int ExtractPlanarFacets(
            ccPointCloud* theCloud,
            unsigned char octreeLevel,
            ScalarType maxError,
            cloudViewer::DistanceComputationTools::ERROR_MEASURES errorMeasure,
            bool useRetroProjectionError = true,
            cloudViewer::GenericProgressCallback* progressCb = 0,
            cloudViewer::DgmOctree* _theOctree = 0);

    //! Default constructor
    FastMarchingForFacetExtraction();

    //! Destructor
    virtual ~FastMarchingForFacetExtraction();

    //! Initializes the grid with a point cloud (and ist corresponding octree)
    /** The points should be associated to an (active) scalar field.
            The Fast Marching grid will have the same dimensions as
            the input octree considered at a given level of subdivision.
            \param cloud the point cloud
            \param theOctree the associated octree
            \param gridLevel the level of subdivision
            \param maxError maximum error allowed by 'propagated' facet
            \param errorMeasure error measure
            \param useRetroProjectionError whether to use retro-projection error
    in propagation \param progressCb progeress callback \return a negative value
    if something went wrong
    **/
    int init(ccGenericPointCloud* cloud,
             cloudViewer::DgmOctree* theOctree,
             unsigned char gridLevel,
             ScalarType maxError,
             cloudViewer::DistanceComputationTools::ERROR_MEASURES errorMeasure,
             bool useRetroProjectionError,
             cloudViewer::GenericProgressCallback* progressCb = 0);

    //! Updates a list of point flags, indicating the points alreay processed
    /** \return the number of newly flagged points
     **/
    unsigned updateFlagsTable(ccGenericPointCloud* theCloud,
                              std::vector<unsigned char>& flags,
                              unsigned facetIndex);

    //! Sets the propagation progress callback
    void setPropagateCallback(
            cloudViewer::GenericProgressCallback* propagateProgressCb) {
        m_propagateProgressCb = propagateProgressCb;
        m_propagateProgress = 0;
    }

    // inherited methods (see FastMarchingAlgorithm)
    virtual int propagate() override;
    virtual bool setSeedCell(const Tuple3i& pos) override;

protected:
    //! A Fast Marching grid cell for planar facets extraction
    class PlanarCell : public cloudViewer::FastMarching::Cell {
    public:
        //! Default constructor
        PlanarCell()
            : Cell(), N(0, 0, 0), C(0, 0, 0), cellCode(0), planarError(0) {}

        ///! Destructor
        virtual ~PlanarCell() {}

        //! The local cell normal
        CCVector3 N;
        //! The local cell center
        CCVector3 C;
        //! the code of the equivalent cell in the octree
        cloudViewer::DgmOctree::CellCode cellCode;
        //! Cell planarity error
        ScalarType planarError;
    };

    // inherited methods (see FastMarchingAlgorithm)
    virtual float computeTCoefApprox(
            cloudViewer::FastMarching::Cell* currentCell,
            cloudViewer::FastMarching::Cell* neighbourCell) const override;
    virtual int step() override;
    virtual void initTrialCells() override;
    virtual bool instantiateGrid(unsigned size) override {
        return instantiateGridTpl<PlanarCell*>(size);
    }

    //! Adds a given cell's points to the current facet and returns the
    //! resulting RMS
    ScalarType addCellToCurrentFacet(unsigned index);

    //! Current facet points
    cloudViewer::ReferenceCloud* m_currentFacetPoints;

    //! Current facet error
    ScalarType m_currentFacetError;

    //! Max facet error
    ScalarType m_maxError;

    //! Error measrue
    cloudViewer::DistanceComputationTools::ERROR_MEASURES m_errorMeasure;

    //! Whether to use retro-projection error in propagation
    bool m_useRetroProjectionError;

    //! Propagation progress callback
    cloudViewer::GenericProgressCallback* m_propagateProgressCb;
    //! Propagation progress
    unsigned m_propagateProgress;
};
