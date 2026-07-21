// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ThreadControlWidget.h"

#include "BundleAdjustmentWidget.h"
#include "util/threading.h"

namespace cloudViewer {

using namespace colmap;

ThreadControlWidget::ThreadControlWidget(QWidget* parent)
    : QWidget(parent),
      progress_bar_(nullptr),
      cancel_action_(new QAction(this)),
      finished_action_(new QAction(this)),
      cleanup_timer_(new QTimer(this)),
      thread_(nullptr) {
    cleanup_timer_->setInterval(200);
    connect(cleanup_timer_, &QTimer::timeout, this,
            &ThreadControlWidget::OnThreadFinished);

    connect(cancel_action_, &QAction::triggered, this,
            &ThreadControlWidget::RequestCancel);
    connect(finished_action_, &QAction::triggered, this,
            &ThreadControlWidget::OnThreadFinished);
}

void ThreadControlWidget::RequestCancel() {
    if (thread_) {
        thread_->Stop();
    }
    if (progress_bar_ != nullptr) {
        progress_bar_->hide();
    }
    // Do not Wait() on the UI thread: some pipeline stages (e.g. Delaunay
    // meshing) run synchronously and cannot exit until they finish.
    cleanup_timer_->start();
}

void ThreadControlWidget::OnThreadFinished() {
    if (!thread_) {
        cleanup_timer_->stop();
        if (progress_bar_ != nullptr) {
            progress_bar_->hide();
        }
        return;
    }

    if (!thread_->IsFinished()) {
        return;
    }

    cleanup_timer_->stop();
    thread_->Wait();
    thread_.reset();
    if (progress_bar_ != nullptr) {
        progress_bar_->hide();
    }
}

void ThreadControlWidget::StartThread(const QString& progress_text,
                                      const bool stoppable,
                                      Thread* thread) {
    CHECK(!thread_);
    CHECK_NOTNULL(thread);

    cleanup_timer_->stop();
    thread_.reset(thread);

    if (progress_bar_ == nullptr) {
        progress_bar_ = new QProgressDialog(this);
        progress_bar_->setWindowModality(Qt::ApplicationModal);
        progress_bar_->setWindowFlags(Qt::Dialog | Qt::WindowTitleHint |
                                      Qt::CustomizeWindowHint);
        // Use a single space to clear the window title on Windows, otherwise it
        // will contain the name of the executable.
        progress_bar_->setWindowTitle(" ");
        progress_bar_->setLabel(new QLabel(this));
        progress_bar_->setMaximum(0);
        progress_bar_->setMinimum(0);
        progress_bar_->setValue(0);
        connect(progress_bar_, &QProgressDialog::canceled, cancel_action_,
                static_cast<void (QAction::*)()>(&QAction::trigger));
    }

    // Enable the cancel button if the thread is stoppable.
    QPushButton* cancel_button =
            progress_bar_->findChildren<QPushButton*>().at(0);
    cancel_button->setEnabled(stoppable);

    progress_bar_->setLabelText(progress_text);

    // Center the progress bar wrt. the parent widget.
    const QPoint global =
            parentWidget()->mapToGlobal(parentWidget()->rect().center());
    progress_bar_->move(global.x() - progress_bar_->width() / 2,
                        global.y() - progress_bar_->height() / 2);

    progress_bar_->show();
    progress_bar_->raise();

    thread_->AddCallback(Thread::FINISHED_CALLBACK,
                         [this]() { finished_action_->trigger(); });
    thread_->Start();
}

void ThreadControlWidget::StartFunction(const QString& progress_text,
                                        const std::function<void()>& func) {
    class FunctionThread : public Thread {
    public:
        explicit FunctionThread(const std::function<void()>& f) : func_(f) {}

    private:
        void Run() { func_(); }
        const std::function<void()> func_;
    };

    StartThread(progress_text, false, new FunctionThread(func));
}

}  // namespace cloudViewer
