// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>

#include <filesystem>
#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/mcalib_tool_common.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: mcalib_static_aruco_detect <dict_id> <input_dir> "
                     "[--interactive]\n";
        return 1;
    }

    const int dict_id = std::stoi(argv[1]);
    const fs::path input_dir = argv[2];
    bool interactive = false;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--interactive") interactive = true;
    }

    cv::Ptr<cv::aruco::Dictionary> dictionary =
            cv::makePtr<cv::aruco::Dictionary>(
                    cv::aruco::getPredefinedDictionary(dict_id));
    cv::Ptr<cv::aruco::DetectorParameters> params =
            cv::makePtr<cv::aruco::DetectorParameters>();

    std::vector<fs::path> images;
    mcalib::tools::collectImages(input_dir, images);
    const fs::path output_dir = input_dir / "result";
    fs::create_directories(output_dir);

    int marker_sum = 0;
    for (const auto& image_path : images) {
        cv::Mat image = cv::imread(image_path.string());
        if (image.empty()) continue;
        if (image.cols > 2000) {
            cv::resize(image, image, cv::Size(), 0.5, 0.5);
        }

        std::vector<int> marker_ids;
        std::vector<std::vector<cv::Point2f>> marker_corners;
        std::vector<std::vector<cv::Point2f>> rejected;
        cv::aruco::detectMarkers(image, dictionary, marker_corners, marker_ids,
                                 params, rejected);
        marker_sum += static_cast<int>(marker_ids.size());

        if (!marker_ids.empty()) {
            cv::aruco::drawDetectedMarkers(image, marker_corners, marker_ids);
            const fs::path out = output_dir / image_path.filename();
            cv::imwrite(out.string(), image);
        } else if (interactive) {
            cv::imshow("image", image);
            if (cv::waitKey(0) == 27) break;
        }
    }

    if (interactive) cv::destroyAllWindows();
    CVLog::Print("[static_aruco] markers=%d images=%zu", marker_sum,
                 images.size());
    return 0;
}
