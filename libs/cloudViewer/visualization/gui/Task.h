// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>

namespace cloudViewer {
namespace visualization {
namespace gui {

class Task {
public:
    /// Runs \param f in another thread. \p f may want to call
    /// Application::PostToMainThread() to communicate the results.
    Task(std::function<void()> f);

    Task(const Task&) = delete;
    Task& operator=(const Task& other) = delete;

    /// Will call WaitToFinish(), which may block.
    ~Task();

    void Run();

    bool IsFinished() const;

    /// This must be called for all tasks eventually or the process will not
    /// exit.
    void WaitToFinish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
