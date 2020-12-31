// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <chrono>
#include <memory>
#include <string>

#include "CloudViewer.h"

using namespace cloudViewer;
namespace tio = cloudViewer::t::io;
namespace sc = std::chrono;


void PrintUsage() {
    PrintCloudViewerVersion();
    // clang-format off
    CVLib::utility::LogInfo(
            "Open a RealSense camera and display live color and depth streams. You can set\n"
            "frame sizes and frame rates for each stream and the depth stream can be\n"
            "optionally aligned to the color stream. NOTE: An error of 'UNKNOWN: Couldn't\n"
            "resolve requests' implies  unsupported stream format settings.");
    // clang-format on
    CVLib::utility::LogInfo("Usage:");
    CVLib::utility::LogInfo(
            "RealSenseRecorder [-h|--help] [-V] [-l|--list-devices] [--align]\n"
            "[--record rgbd_video_file.bag] [-c|--config rs-config.json]");
}

int main(int argc, char **argv) {
    // Parse command line arguments
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--help") ||
        CVLib::utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintUsage();
        return 0;
    }
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--list-devices") ||
        CVLib::utility::ProgramOptionExists(argc, argv, "-l")) {
        tio::RealSenseSensor::ListDevices();
        return 0;
    }
    if (CVLib::utility::ProgramOptionExists(argc, argv, "-V")) {
        CVLib::utility::SetVerbosityLevel(CVLib::utility::VerbosityLevel::Debug);
    } else {
        CVLib::utility::SetVerbosityLevel(CVLib::utility::VerbosityLevel::Info);
    }
    bool align_streams = false;
    std::string config_file, bag_file;

    if (CVLib::utility::ProgramOptionExists(argc, argv, "-c")) {
        config_file = CVLib::utility::GetProgramOptionAsString(argc, argv, "-c");
    } else if (CVLib::utility::ProgramOptionExists(argc, argv, "--config")) {
        config_file = CVLib::utility::GetProgramOptionAsString(argc, argv, "--config");
    } else {
        CVLib::utility::LogError("config json file required.");
        PrintUsage();
        return 1;
    }
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--align")) {
        align_streams = true;
    }
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--record")) {
        bag_file = CVLib::utility::GetProgramOptionAsString(argc, argv, "--record");
    }

    // Read in camera configuration.
    tio::RealSenseSensorConfig rs_cfg;
    cloudViewer::io::ReadIJsonConvertible(config_file, rs_cfg);

    // Initialize camera.
    tio::RealSenseSensor rs;
    rs.ListDevices();
    rs.InitSensor(rs_cfg, 0, bag_file);
    CVLib::utility::LogInfo("{}", rs.GetMetadata().ToString());

    // Create windows to show depth and color streams.
    bool flag_start = false, flag_record = flag_start, flag_exit = false;
    visualization::VisualizerWithKeyCallback depth_vis, color_vis;
    auto callback_exit = [&](visualization::Visualizer *vis) {
        flag_exit = true;
        if (flag_start) {
            CVLib::utility::LogInfo("Recording finished.");
        } else {
            CVLib::utility::LogInfo("Nothing has been recorded.");
        }
        return false;
    };
    depth_vis.RegisterKeyCallback(GLFW_KEY_ESCAPE, callback_exit);
    color_vis.RegisterKeyCallback(GLFW_KEY_ESCAPE, callback_exit);
    auto callback_toggle_record = [&](visualization::Visualizer *vis) {
        if (flag_record) {
            rs.PauseRecord();
            CVLib::utility::LogInfo(
                    "Recording paused. "
                    "Press [SPACE] to continue. "
                    "Press [ESC] to save and exit.");
            flag_record = false;
        } else {
            rs.ResumeRecord();
            flag_record = true;
            if (!flag_start) {
                CVLib::utility::LogInfo(
                        "Recording started. "
                        "Press [SPACE] to pause. "
                        "Press [ESC] to save and exit.");
                flag_start = true;
            } else {
                CVLib::utility::LogInfo(
                        "Recording resumed, video may be discontinuous. "
                        "Press [SPACE] to pause. "
                        "Press [ESC] to save and exit.");
            }
        }
        return false;
    };
    if (!bag_file.empty()) {
        depth_vis.RegisterKeyCallback(GLFW_KEY_SPACE, callback_toggle_record);
        color_vis.RegisterKeyCallback(GLFW_KEY_SPACE, callback_toggle_record);
        CVLib::utility::LogInfo(
                "In the visulizer window, "
                "press [SPACE] to start recording, "
                "press [ESC] to exit.");
    } else {
        CVLib::utility::LogInfo("In the visulizer window, press [ESC] to exit.");
    }

    using legacyRGBDImage = cloudViewer::geometry::RGBDImage;
    using legacyImage = cloudViewer::geometry::Image;
    std::shared_ptr<legacyImage> depth_image_ptr, color_image_ptr;

    // Loop over frames from device
    legacyRGBDImage im_rgbd;
    bool is_geometry_added = false;
    size_t frame_id = 0;
    rs.StartCapture(flag_start);
    do {
        im_rgbd = rs.CaptureFrame(true, align_streams).ToLegacyRGBDImage();

        // Improve depth visualization by scaling
        /* im_rgbd.depth_.LinearTransform(0.25); */
        depth_image_ptr = std::shared_ptr<cloudViewer::geometry::Image>(
                &im_rgbd.depth_, [](cloudViewer::geometry::Image *) {});
        color_image_ptr = std::shared_ptr<cloudViewer::geometry::Image>(
                &im_rgbd.color_, [](cloudViewer::geometry::Image *) {});

        if (!is_geometry_added) {
            if (!depth_vis.CreateVisualizerWindow(
                        "cloudViewer || RealSense || Depth", depth_image_ptr->width_,
                        depth_image_ptr->height_, 15, 50) ||
                !depth_vis.AddGeometry(depth_image_ptr) ||
                !color_vis.CreateVisualizerWindow(
                        "cloudViewer || RealSense || Color", color_image_ptr->width_,
                        color_image_ptr->height_, 675, 50) ||
                !color_vis.AddGeometry(color_image_ptr)) {
                CVLib::utility::LogError("Window creation failed!");
                return 0;
            }
            is_geometry_added = true;
        }

        depth_vis.UpdateGeometry();
        color_vis.UpdateGeometry();
        depth_vis.PollEvents();
        color_vis.PollEvents();
        depth_vis.UpdateRender();
        color_vis.UpdateRender();

        if (frame_id++ % 30 == 0) {
            CVLib::utility::LogInfo("Time: {}s, Frame {}",
                             static_cast<double>(rs.GetTimestamp()) * 1e-6,
                             frame_id - 1);
        }
    } while (!flag_exit);

    rs.StopCapture();
    return 0;
}
