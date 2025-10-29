// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <chrono>

namespace colmap {

class Timer {
public:
    Timer();

    void Start();
    void Restart();
    void Pause();
    void Resume();
    void Reset();

    double ElapsedMicroSeconds() const;
    double ElapsedSeconds() const;
    double ElapsedMinutes() const;
    double ElapsedHours() const;

    void PrintSeconds() const;
    void PrintMinutes() const;
    void PrintHours() const;

private:
    bool started_;
    bool paused_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point pause_time_;
};

}  // namespace colmap
