// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVLog.h>

#include <filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/mcalib_tool_common.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: mcalib_static_chessboard_detect <input_dir> "
                     "[cols rows] [--interactive]\n";
        return 1;
    }

    const fs::path input_dir = argv[1];
    cv::Size board_size(4, 4);
    bool interactive = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--interactive") {
            interactive = true;
        } else if (board_size.width == 4 && board_size.height == 4) {
            board_size.width = std::stoi(arg);
            if (i + 1 < argc && std::string(argv[i + 1]) != "--interactive") {
                board_size.height = std::stoi(argv[++i]);
            }
        }
    }

    std::vector<fs::path> images;
    mcalib::tools::collectImages(input_dir, images);
    const fs::path output_dir = input_dir / "chessboard_result";
    fs::create_directories(output_dir);

    int found = 0;
    for (const auto& image_path : images) {
        cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;

        std::vector<cv::Point2f> corners;
        const bool ok = cv::findChessboardCorners(
                image, board_size, corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (!ok) continue;
        ++found;

        cv::Mat color;
        cv::cvtColor(image, color, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(color, board_size, corners, ok);
        cv::imwrite((output_dir / image_path.filename()).string(), color);

        if (interactive) {
            cv::imshow("chessboard", color);
            if (cv::waitKey(0) == 27) break;
        }
    }

    if (interactive) cv::destroyAllWindows();
    CVLog::Print("[static_chessboard] found=%d images=%zu", found,
                 images.size());
    return 0;
}
