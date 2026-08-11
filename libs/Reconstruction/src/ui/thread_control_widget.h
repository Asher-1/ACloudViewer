// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "util/threading.h"

namespace colmap {

class ThreadControlWidget : public QWidget {
public:
    explicit ThreadControlWidget(QWidget* parent);

    void StartThread(const QString& progress_text,
                     const bool stoppable,
                     Thread* thread);
    void StartFunction(const QString& progress_text,
                       const std::function<void()>& func);

private:
    void RequestCancel();
    void OnThreadFinished();

    QProgressDialog* progress_bar_;
    QAction* cancel_action_;
    QAction* finished_action_;
    QTimer* cleanup_timer_;
    std::unique_ptr<Thread> thread_;
};

}  // namespace colmap
