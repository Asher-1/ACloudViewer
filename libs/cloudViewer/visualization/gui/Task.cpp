// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "visualization/gui/Task.h"

#include <Logging.h>

#include <atomic>
#include <thread>

namespace cloudViewer {
namespace visualization {
namespace gui {

namespace {
enum class ThreadState { NOT_STARTED, RUNNING, FINISHED };
}

struct Task::Impl {
    std::function<void()> func_;
    std::thread thread_;
    ThreadState state_;
    std::atomic<bool> is_finished_running_;
};

Task::Task(std::function<void()> f) : impl_(new Task::Impl) {
    impl_->func_ = f;
    impl_->state_ = ThreadState::NOT_STARTED;
    impl_->is_finished_running_ = false;
}

Task::~Task() {
    // TODO: if able to cancel, do so here
    WaitToFinish();
}

void Task::Run() {
    if (impl_->state_ != ThreadState::NOT_STARTED) {
        utility::LogWarning("Attempting to Run() already-started Task");
        return;
    }

    auto thread_main = [this]() {
        impl_->func_();
        impl_->is_finished_running_ = true;
    };
    impl_->thread_ = std::thread(thread_main);  // starts thread
    impl_->state_ = ThreadState::RUNNING;
}

bool Task::IsFinished() const {
    switch (impl_->state_) {
        case ThreadState::NOT_STARTED:
            return true;
        case ThreadState::RUNNING:
            return impl_->is_finished_running_;
        case ThreadState::FINISHED:
            return true;
    }
    utility::LogError("Unexpected thread state");
}

void Task::WaitToFinish() {
    if (impl_->state_ == ThreadState::RUNNING) {
        impl_->thread_.join();
        impl_->state_ = ThreadState::FINISHED;
    }
}

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
