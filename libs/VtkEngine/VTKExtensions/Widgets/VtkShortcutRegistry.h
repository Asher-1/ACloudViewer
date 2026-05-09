// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file VtkShortcutRegistry.h
 * @brief Shared VTK shortcut definitions used by both the shortcut dialog
 *        (editing / persistence) and QVTKWidgetCustom (runtime dispatch).
 *
 * Each entry maps a human-readable action to the parameters that
 * vtkCustomInteractorStyle::handleShortcut() expects.  Users may remap
 * shortcuts; the remapped QKeySequence is persisted in QSettings under
 * the "VtkShortcuts" group while the underlying VTK action parameters
 * remain constant.
 */

#include <QKeySequence>
#include <QMap>
#include <QSettings>
#include <QString>
#include <QVector>

struct VtkShortcutDef {
    QString id;
    QString label;
    QKeySequence defaultKey;
    char vtkKey;
    bool vtkCtrl;
    bool vtkAlt;
    bool vtkShift;
};

inline QVector<VtkShortcutDef> vtkDefaultShortcuts() {
    return {
            // Ctrl+Alt shortcuts
            {"vtk_screenshot", "VTK: Screenshot",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_J), 'j', true, true,
             false},
            {"vtk_camera_info", "VTK: Camera Info",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_C), 'c', true, true,
             false},
            {"vtk_toggle_grid", "VTK: Toggle Grid",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_G), 'g', true, true,
             false},
            {"vtk_toggle_lut", "VTK: Toggle LUT",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_K), 'k', true, true,
             false},
            {"vtk_toggle_proj", "VTK: Toggle Parallel Proj",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_O), 'o', true, true,
             false},
            {"vtk_fullscreen", "VTK: Toggle Fullscreen",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_F), 'f', true, true,
             false},
            {"vtk_stereo", "VTK: Toggle Stereo",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_S), 's', true, true,
             false},
            {"vtk_zoom_in", "VTK: Zoom In",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_Plus), '+', true, true,
             false},
            {"vtk_zoom_out", "VTK: Zoom Out",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_Minus), '-', true, true,
             false},

            // Ctrl+Shift shortcuts
            {"vtk_points_mode", "VTK: Points Mode",
             QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_D), 'p', true, false,
             true},
            {"vtk_wireframe_mode", "VTK: Wireframe Mode",
             QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_W), 'w', true, false,
             true},
            {"vtk_surface_mode", "VTK: Surface Mode",
             QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_F), 's', true, false,
             true},
            {"vtk_ptsize_inc", "VTK: Increase Point Size",
             QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Plus), '+', true,
             false, true},
            {"vtk_ptsize_dec", "VTK: Decrease Point Size",
             QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Minus), '-', true,
             false, true},

            // Ctrl-only shortcuts
            {"vtk_save_camera", "VTK: Save Camera",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_M), 'm', true, true,
             false},
            {"vtk_restore_camera", "VTK: Restore Camera",
             QKeySequence(Qt::CTRL | Qt::Key_R), 'r', true, false, false},

            // View presets (Ctrl+Alt+1..6)
            {"vtk_view_front", "VTK: Front View (+Z)",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_1), '1', true, true,
             false},
            {"vtk_view_back", "VTK: Back View (-Z)",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_2), '2', true, true,
             false},
            {"vtk_view_left", "VTK: Left View (-X)",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_3), '3', true, true,
             false},
            {"vtk_view_right", "VTK: Right View (+X)",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_4), '4', true, true,
             false},
            {"vtk_view_top", "VTK: Top View (+Y)",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_5), '5', true, true,
             false},
            {"vtk_view_bottom", "VTK: Bottom View (-Y)",
             QKeySequence(Qt::CTRL | Qt::ALT | Qt::Key_6), '6', true, true,
             false},
    };
}

/// Build the runtime dispatch map: QKeySequence (possibly remapped) →
/// VtkShortcutDef. Reads overrides from QSettings("VtkShortcuts").
inline QMap<QString, VtkShortcutDef> buildVtkShortcutMap() {
    QMap<QString, VtkShortcutDef> map;

    QSettings settings;
    settings.beginGroup(QStringLiteral("VtkShortcuts"));

    for (const auto& def : vtkDefaultShortcuts()) {
        QKeySequence seq = def.defaultKey;
        if (settings.contains(def.id)) {
            seq = settings.value(def.id).value<QKeySequence>();
        }
        if (!seq.isEmpty()) {
            VtkShortcutDef entry = def;
            entry.defaultKey = seq;
            map[seq.toString(QKeySequence::PortableText)] = entry;
        }
    }

    settings.endGroup();
    return map;
}
