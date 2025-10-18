// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

namespace colmap {
class Thread;
}

namespace cloudViewer {

class ThreadControlWidget : public QWidget {
public:
    explicit ThreadControlWidget(QWidget* parent);

    void StartThread(const QString& progress_text,
                     const bool stoppable,
                     colmap::Thread* thread);
    void StartFunction(const QString& progress_text,
                       const std::function<void()>& func);

private:
    QProgressDialog* progress_bar_;
    QAction* destructor_;
    std::unique_ptr<colmap::Thread> thread_;
};

}  // namespace cloudViewer
