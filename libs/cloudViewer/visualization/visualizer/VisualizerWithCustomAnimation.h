// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "visualization/visualizer/Visualizer.h"

namespace cloudViewer {
namespace visualization {

class VisualizerWithCustomAnimation : public Visualizer {
public:

    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    VisualizerWithCustomAnimation();
    ~VisualizerWithCustomAnimation() override;
    VisualizerWithCustomAnimation(const VisualizerWithCustomAnimation &) =
            delete;
    VisualizerWithCustomAnimation &operator=(
            const VisualizerWithCustomAnimation &) = delete;

public:
    void PrintVisualizerHelp() override;
    void UpdateWindowTitle() override;
    void Play(bool recording = false,
              bool recording_depth = false,
              bool close_window_when_animation_ends = false);
    void RegisterRecordingImageFormat(const std::string &basedir,
                                      const std::string &format,
                                      const std::string &trajectory) {
        recording_image_basedir_ = basedir;
        recording_image_filename_format_ = format;
        recording_image_trajectory_filename_ = trajectory;
    }
    void RegisterRecordingDepthFormat(const std::string &basedir,
                                      const std::string &format,
                                      const std::string &trajectory) {
        recording_depth_basedir_ = basedir;
        recording_depth_filename_format_ = format;
        recording_depth_trajectory_filename_ = trajectory;
    }

protected:
    bool InitViewControl() override;
    void MouseMoveCallback(GLFWwindow *window, double x, double y) override;
    void MouseScrollCallback(GLFWwindow *window, double x, double y) override;
    void MouseButtonCallback(GLFWwindow *window,
                             int button,
                             int action,
                             int mods) override;
    void KeyPressCallback(GLFWwindow *window,
                          int key,
                          int scancode,
                          int action,
                          int mods) override;

protected:
    std::string recording_image_basedir_ = "image/";
    std::string recording_image_filename_format_ = "image_{:06d}.png";
    std::string recording_image_trajectory_filename_ = "image_trajectory.json";
    std::string recording_depth_basedir_ = "depth/";
    std::string recording_depth_filename_format_ = "depth_{:06d}.png";
    std::string recording_depth_trajectory_filename_ = "depth_trajectory.json";
    size_t recording_file_index_ = 0;
};

}  // namespace visualization
}  // namespace cloudViewer
