// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PYTHON_PLUGIN_EDITOR_SETTINGS_H
#define PYTHON_PLUGIN_EDITOR_SETTINGS_H

#include <ui_EditorSettings.h>

#include "ColorScheme.h"

class EditorSettings final : public QDialog, public Ui::EditorSettings
{
    Q_OBJECT
  public:
    EditorSettings();

    int fontSize() const;
    bool shouldHighlightCurrentLine() const;
    const ColorScheme &colorScheme() const;

  public:
  Q_SIGNALS:
    void settingsChanged();

  protected:
    void connectSignals() const;
    void saveChanges();
    void saveChangesAndClose();
    void cancel();
    void setFormValuesToCurrentValues() const;
    void loadFromSettings();
    void setupUi();

  private:
    ColorScheme m_colorScheme = ColorScheme::Default();
    bool m_shouldHighlightCurrentLine = true;
};

#endif // PYTHON_PLUGIN_EDITOR_SETTINGS_H
