// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

class ccPointCloud;
class ecvProgressDialog;

//! Minimum Spanning Tree for normals direction resolution
/** See http://people.maths.ox.ac.uk/wendland/research/old/reconhtml/node3.html
 **/
class ccMinimumSpanningTreeForNormsDirection {
public:
    //! Main entry point
    static bool OrientNormals(ccPointCloud* cloud,
                              unsigned kNN = 6,
                              ecvProgressDialog* progressDlg = 0);
};
