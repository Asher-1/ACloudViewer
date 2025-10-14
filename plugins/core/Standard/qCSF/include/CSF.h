// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "Cloth.h"
#include "wlPointCloud.h"

// system
#include <string>
#include <vector>

class ecvMainAppInterface;
class QWidget;
class ccMesh;

class CSF {
public:
    CSF(wl::PointCloud& cloud);
    virtual ~CSF();

    // input PC from files
    bool readPointsFromFile(std::string filename);

    // save the ground points to file
    void saveGroundPoints(const std::vector<int>& grp, std::string path = "");
    void saveOffGroundPoints(const std::vector<int>& grp,
                             std::string path = "");

    // The main program: Do filtering
    bool do_filtering(std::vector<int>& groundIndexes,
                      std::vector<int>& offGroundIndexes,
                      bool exportClothMesh,
                      ccMesh*& clothMesh,
                      ecvMainAppInterface* app = 0,
                      QWidget* parent = 0);

private:
    wl::PointCloud& point_cloud;

public:
    struct Parameters {
        // parameters
        int k_nearest_points;

        bool bSloopSmooth;

        double time_step;

        double class_threshold;

        double cloth_resolution;

        int rigidness;

        int iterations;
    };

    Parameters params;
};
