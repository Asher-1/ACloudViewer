// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QtCore>
#include <QtWidgets>

#include "util/misc.h"
#include "util/option_manager.h"

namespace colmap {

class ProjectWidget : public QWidget {
public:
    ProjectWidget(QWidget* parent, OptionManager* options);

    bool IsValid() const;
    void Reset();

    std::string GetDatabasePath() const;
    std::string GetImagePath() const;
    void SetDatabasePath(const std::string& path);
    void SetImagePath(const std::string& path);

private:
    void Save();
    void SelectNewDatabasePath();
    void SelectExistingDatabasePath();
    void SelectImagePath();
    QString DefaultDirectory();

    OptionManager* options_;

    // Whether file dialog was opened previously.
    bool prev_selected_;

    // Text boxes that hold the currently selected paths.
    QLineEdit* database_path_text_;
    QLineEdit* image_path_text_;
};

}  // namespace colmap
