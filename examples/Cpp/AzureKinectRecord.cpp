// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <k4a/k4a.h>

#include "assert.h"

#include <math.h>
#include <atomic>
#include <csignal>
#include <ctime>
#include <iostream>

#include "CloudViewer.h"

using namespace cloudViewer;

void PrintUsage() {
    PrintCloudViewerVersion();
    // clang-format off
    cloudViewer::utility::LogInfo("Usage: ");
    cloudViewer::utility::LogInfo("Options: ");
    cloudViewer::utility::LogInfo("--config  Config .json file (default: none)");
    cloudViewer::utility::LogInfo("--list    List the currently connected K4A devices");
    cloudViewer::utility::LogInfo("--device  Specify the device index to use (default: 0)");
    cloudViewer::utility::LogInfo("--output  Output mkv file name (default: current_time.mkv)");
    cloudViewer::utility::LogInfo("-a        Align depth with color image (default: disabled)");
    cloudViewer::utility::LogInfo("-h        Print this helper");
    // clang-format on
}

int main(int argc, char **argv) {
    if (argc < 2) {
        PrintUsage();
        return 0;
    }

    // Parse arguments
    if (cloudViewer::utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintUsage();
        return 0;
    }

    if (cloudViewer::utility::ProgramOptionExists(argc, argv, "--list")) {
        io::AzureKinectSensor::ListDevices();
        return 0;
    }

    io::AzureKinectSensorConfig sensor_config;
    if (cloudViewer::utility::ProgramOptionExists(argc, argv, "--config")) {
        auto config_filename =
                cloudViewer::utility::GetProgramOptionAsString(argc, argv, "--config", "");
        bool success = io::ReadIJsonConvertibleFromJSON(config_filename,
                                                        sensor_config);
        if (!success) {
            cloudViewer::utility::LogInfo("Invalid sensor config");
            return 1;
        }
    } else {
        cloudViewer::utility::LogInfo("Use default sensor config");
    }

    int sensor_index =
            cloudViewer::utility::GetProgramOptionAsInt(argc, argv, "--device", 0);
    if (sensor_index < 0 || sensor_index > 255) {
        cloudViewer::utility::LogWarning("Sensor index must between [0, 255]: {}",
                            sensor_index);
        return 1;
    }

    bool enable_align_depth_to_color =
            cloudViewer::utility::ProgramOptionExists(argc, argv, "-a");

    std::string recording_filename = cloudViewer::utility::GetProgramOptionAsString(
            argc, argv, "--output", cloudViewer::utility::GetCurrentTimeStamp() + ".mkv");
    cloudViewer::utility::LogInfo("Prepare writing to {}", recording_filename);

    // Init recorder
    io::AzureKinectRecorder recorder(sensor_config, sensor_index);
    if (!recorder.InitSensor()) {
        cloudViewer::utility::LogWarning("Failed to connect to sensor, abort.");
        return 1;
    }

    bool flag_record = false;
    bool flag_exit = false;
    bool is_geometry_added = false;
    visualization::VisualizerWithKeyCallback vis;
    vis.RegisterKeyCallback(
            GLFW_KEY_SPACE, [&](visualization::Visualizer *vis) {
                if (flag_record) {
                    cloudViewer::utility::LogInfo(
                            "Recording paused. "
                            "Press [SPACE] to continue. "
                            "Press [ESC] to save and exit.");
                    flag_record = false;
                } else if (!recorder.IsRecordCreated()) {
                    if (recorder.OpenRecord(recording_filename)) {
                        cloudViewer::utility::LogInfo(
                                "Recording started. "
                                "Press [SPACE] to pause. "
                                "Press [ESC] to save and exit.");
                        flag_record = true;
                    }  // else flag_record keeps false
                } else {
                    cloudViewer::utility::LogInfo(
                            "Recording resumed, video may be discontinuous. "
                            "Press [SPACE] to pause. "
                            "Press [ESC] to save and exit.");
                    flag_record = true;
                }
                return false;
            });

    vis.RegisterKeyCallback(
            GLFW_KEY_ESCAPE, [&](visualization::Visualizer *vis) {
                flag_exit = true;
                if (recorder.IsRecordCreated()) {
                    cloudViewer::utility::LogInfo("Recording finished.");
                } else {
                    cloudViewer::utility::LogInfo("Nothing has been recorded.");
                }
                return false;
            });

    cloudViewer::utility::LogInfo(
            "In the visulizer window, "
            "press [SPACE] to start recording, "
            "press [ESC] to exit.");

    vis.CreateVisualizerWindow("cloudViewer Azure Kinect Recorder", 1920, 540);
    do {
        auto im_rgbd =
                recorder.RecordFrame(flag_record, enable_align_depth_to_color);
        if (im_rgbd == nullptr) {
            cloudViewer::utility::LogInfo("Invalid capture, skipping this frame");
            continue;
        }

        if (!is_geometry_added) {
            vis.AddGeometry(im_rgbd);
            is_geometry_added = true;
        }

        // Update visualizer
        vis.UpdateGeometry();
        vis.PollEvents();
        vis.UpdateRender();

    } while (!flag_exit);

    recorder.CloseRecord();

    return 0;
}
