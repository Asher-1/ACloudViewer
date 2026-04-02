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
        int k_nearest_points = 1;
        bool bSloopSmooth = true;
        double time_step = 0.65;
        double class_threshold = 0.5;
        double cloth_resolution = 1.0;
        int rigidness = 3;
        int iterations = 500;

        // constants (matching CloudCompare)
        static constexpr double clothYHeight = 0.05;  // origin cloth height
        static constexpr int clothBuffer =
                2;  // cloth buffer (grid margin size)
        static constexpr double gravity = 0.2;
    };

    Parameters params;
};
