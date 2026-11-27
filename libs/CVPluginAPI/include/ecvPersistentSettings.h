// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFileInfo>
#include <QSettings>
#include <QString>

#include "CVPluginAPI.h"

//! Persistent settings keys and QSettings helpers (app + plugins).
//! No class-wide export macro: MSVC would mark the inline key helpers
//! dllimport and fail to resolve them at link time. Only the out-of-line
//! members below carry CVPLUGIN_LIB_API.
class ecvPS {
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
    static inline const QString PickRotationCenter() {
        return QStringLiteral("PickRotationCenter");
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

    /** Register a QSettings group prefix (e.g. "qDA3") for
     * resetAllRegistered(). */
    static CVPLUGIN_LIB_API void registerSettingsGroup(
            const QString& groupPrefix);

    /** Clear every group registered via registerSettingsGroup(). */
    static CVPLUGIN_LIB_API void resetAllRegistered();

    static inline QString settingsKey(const QString& groupPrefix,
                                      const QString& key) {
        return groupPrefix + QLatin1Char('/') + key;
    }

    static inline QString browseDir(QSettings& settings,
                                    const QString& groupPrefix,
                                    const QString& key,
                                    const QString& fallback) {
        return settings.value(settingsKey(groupPrefix, key), fallback)
                .toString();
    }

    static inline void saveBrowseDir(QSettings& settings,
                                     const QString& groupPrefix,
                                     const QString& key,
                                     const QString& fileOrDirPath) {
        if (fileOrDirPath.isEmpty()) return;
        settings.setValue(settingsKey(groupPrefix, key),
                          QFileInfo(fileOrDirPath).absolutePath());
    }

    static inline QString browseFile(QSettings& settings,
                                     const QString& groupPrefix,
                                     const QString& key,
                                     const QString& fallback) {
        return settings.value(settingsKey(groupPrefix, key), fallback)
                .toString();
    }

    static inline void saveBrowseFile(QSettings& settings,
                                      const QString& groupPrefix,
                                      const QString& key,
                                      const QString& filePath) {
        if (filePath.isEmpty()) return;
        settings.setValue(settingsKey(groupPrefix, key),
                          QFileInfo(filePath).absoluteFilePath());
    }
};
