// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CV_DB_LIB
#include <ecvBox.h>
#include <ecvGenericPointCloud.h>

//! Finds a the biggets enclosed rectangle in a point cloud (2D)
class ccInnerRect2DFinder {
public:
    //! Default constructor
    ccInnerRect2DFinder();

    //! Finds the biggest enclosed rectangle
    ccBox* process(ccGenericPointCloud* cloud, unsigned char zDim = 2);

protected:
    //! Initializes internal structures
    bool init(ccGenericPointCloud* cloud, unsigned char zDim);

    //! 2D rectangle
    struct Rect {
        Rect() : x0(0), y0(0), x1(0), y1(0) {}
        Rect(double _x0, double _y0, double _x1, double _y1)
            : x0(_x0), y0(_y0), x1(_x1), y1(_y1) {}

        double x0, y0, x1, y1;

        inline double width() const { return x1 - x0; }
        inline double height() const { return y1 - y0; }
        inline double area() const { return width() * height(); }
    };

    //! Internal processs
    void findBiggestRect(const Rect& rect, unsigned startIndex);

    //! Global rectangle
    Rect m_boundingRect;

    //! Inner rectangle
    Rect m_maxRect;
    //! Inner rectangle max area
    double m_maxArea;

    //! Associated cloud
    ccGenericPointCloud* m_cloud;

    //! X dimension
    unsigned char m_X;
    //! Y dimension
    unsigned char m_Y;
};
