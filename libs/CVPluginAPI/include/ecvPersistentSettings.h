// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVPluginAPI.h"

// Qt
#include <QString>

//! Persistent settings key (to be used with QSettings)
class CVPLUGIN_LIB_API ecvPS {
public:
    static inline const QString LoadFile() {
        return QStringLiteral("LoadFile");
    }
    static inline const QString SaveFile() {
        return QStringLiteral("SaveFile");
    }
    static inline const QString ThemeSettings() {
        return QStringLiteral("ThemeSettings");
    }
    static inline const QString CurrentTheme() {
        return QStringLiteral("CurrentTheme");
    }
    static inline const QString MainWindowSettings() {
        return QStringLiteral("MainWindowSettings");
    }
    static inline const QString MainWinGeom() {
        return QStringLiteral("mainWindowGeometry");
    }
    static inline const QString MainWinState() {
        return QStringLiteral("mainWindowState");
    }
    static inline const QString CurrentPath() {
        return QStringLiteral("currentPath");
    }
    static inline const QString TextureFilePath() {
        return QStringLiteral("TextureFilePath");
    }
    static inline const QString SelectedImageInputFilter() {
        return QStringLiteral("selectedImageInputFilter");
    }
    static inline const QString SelectedInputFilter() {
        return QStringLiteral("selectedInputFilter");
    }
    static inline const QString SelectedOutputFilterCloud() {
        return QStringLiteral("selectedOutputFilterCloud");
    }
    static inline const QString SelectedOutputFilterMesh() {
        return QStringLiteral("selectedOutputFilterMesh");
    }
    static inline const QString SelectedOutputFilterImage() {
        return QStringLiteral("selectedOutputFilterImage");
    }
    static inline const QString SelectedOutputFilterPoly() {
        return QStringLiteral("selectedOutputFilterPoly");
    }
    static inline const QString DuplicatePointsGroup() {
        return QStringLiteral("duplicatePoints");
    }
    static inline const QString DuplicatePointsMinDist() {
        return QStringLiteral("minDist");
    }
    static inline const QString HeightGridGeneration() {
        return QStringLiteral("HeightGridGeneration");
    }
    static inline const QString VolumeCalculation() {
        return QStringLiteral("VolumeCalculation");
    }
    static inline const QString Console() { return QStringLiteral("Console"); }
    static inline const QString GlobalShift() {
        return QStringLiteral("GlobalShift");
    }
    static inline const QString MaxAbsCoord() {
        return QStringLiteral("MaxAbsCoord");
    }
    static inline const QString MaxAbsDiag() {
        return QStringLiteral("MaxAbsDiag");
    }
    static inline const QString AutoPickRotationCenter() {
        return QStringLiteral("AutoPickRotationCenter");
    }
    static inline const QString AutoShowCenter() {
        return QStringLiteral("AutoShowCenter");
    }
    static inline const QString AutoShowReconstructionToolBar() {
        return QStringLiteral("ReconstructionToolBar");
    }
    static inline const QString CustomLayoutGeom() {
        return QStringLiteral("customLayoutGeometry");
    }
    static inline const QString CustomLayoutState() {
        return QStringLiteral("customLayoutState");
    }
    static inline const QString DoNotRestoreWindowGeometry() {
        return QStringLiteral("DoNotRestoreWindowGeometry");
    }
    static inline const QString Shortcuts() {
        return QStringLiteral("Shortcuts");
    }
    static inline const QString AppStyle() {
        return QStringLiteral("AppStyle");
    }
};
