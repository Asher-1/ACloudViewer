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

namespace cloudViewer {

class OptionManager;

class ProjectWidget : public QWidget {
public:
    ProjectWidget(QWidget* parent, OptionManager* options);

    bool IsValid() const;
    void Reset();

    void persistSave(const std::string& project_path,
                     const std::string& database_path,
                     const std::string& image_path);

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

}  // namespace cloudViewer
