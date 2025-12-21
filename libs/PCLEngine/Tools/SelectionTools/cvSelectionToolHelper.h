// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_SELECTION_TOOL_HELPER_H
#define CV_SELECTION_TOOL_HELPER_H

#include <QMessageBox>
#include <QString>
#include <QWidget>

//! Helper utilities for selection tools (ParaView-style)
class cvSelectionToolHelper {
public:
    //! Shows instruction dialog if not disabled by user (ParaView-style)
    /** Similar to pqCoreUtilities::promptUser in ParaView.
     *  \param settingsKey Unique key for storing "don't show again" preference
     *  \param title Dialog title
     *  \param message Dialog message (can use HTML)
     *  \param parent Parent widget
     *  \return True if dialog was shown, false if user has disabled it
     */
    static bool promptUser(const QString& settingsKey,
                           const QString& title,
                           const QString& message,
                           QWidget* parent = nullptr);
};

#endif  // CV_SELECTION_TOOL_HELPER_H
