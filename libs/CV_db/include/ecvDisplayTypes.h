// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFont>
#include <QFontMetrics>
#include <QList>
#include <QPair>
#include <QPoint>
#include <QRect>
#include <QString>
#include <QWidget>
#include <list>
#include <vector>

#include "CV_db.h"
#include "ecvGenericGLDisplay.h"

// ============================================================================
// Display-related POD types extracted from ecvDisplayTools
//
// These types were originally nested inside ecvDisplayTools.  They are now
// free-standing so that VtkEngine (and other consumers) can use them without
// pulling in the full ecvDisplayTools header.
// ============================================================================

struct CV_DB_LIB_API ecvMessageToDisplay {
    ecvMessageToDisplay()
        : messageValidity_sec(0),
          position(ecvGenericGLDisplay::LOWER_LEFT_MESSAGE),
          type(ecvGenericGLDisplay::CUSTOM_MESSAGE) {}

    QString message;
    qint64 messageValidity_sec;
    ecvGenericGLDisplay::MessagePosition position;
    ecvGenericGLDisplay::MessageType type;
};

struct CV_DB_LIB_API ecvProjectionMetrics {
    ecvProjectionMetrics()
        : zNear(0.0), zFar(0.0), cameraToBBCenterDist(0.0), bbHalfDiag(0.0) {}

    double zNear;
    double zFar;
    double cameraToBBCenterDist;
    double bbHalfDiag;
};

struct CV_DB_LIB_API ecvHotZone {
    QFont font;
    int textHeight;
    int yTextBottomLineShift;
    unsigned char color[3];

    QString bbv_label;
    QRect bbv_labelRect;
    int bbv_totalWidth;

    QString fs_label;
    QRect fs_labelRect;
    int fs_totalWidth;

    QString psi_label;
    QRect psi_labelRect;
    int psi_totalWidth;

    QString lsi_label;
    QRect lsi_labelRect;
    int lsi_totalWidth;

    int margin;
    int iconSize;
    QPoint topCorner;
    qreal pixelDeviceRatio;

    explicit ecvHotZone(QWidget* win)
        : textHeight(0),
          yTextBottomLineShift(0),
          bbv_label("bubble-view mode"),
          fs_label("fullscreen mode"),
          psi_label("default point size"),
          lsi_label("default line width"),
          margin(16),
          iconSize(16),
          topCorner(0, 0),
          pixelDeviceRatio(1.0) {
        color[0] = ecvColor::defaultLabelBkgColor.r;
        color[1] = ecvColor::defaultLabelBkgColor.g;
        color[2] = ecvColor::defaultLabelBkgColor.b;

        updateInternalVariables(win);
    }

    void updateInternalVariables(QWidget* win);

    QRect rect(bool clickableItemsVisible,
               bool bubbleViewModeEnabled,
               bool fullScreenEnabled) const;
};

struct ecvClickableItem {
    enum Role {
        NO_ROLE,
        INCREASE_POINT_SIZE,
        DECREASE_POINT_SIZE,
        INCREASE_LINE_WIDTH,
        DECREASE_LINE_WIDTH,
        LEAVE_BUBBLE_VIEW_MODE,
        LEAVE_FULLSCREEN_MODE,
    };

    ecvClickableItem() : role(NO_ROLE) {}
    ecvClickableItem(Role _role, QRect _area) : role(_role), area(_area) {}

    Role role;
    QRect area;
};

struct ecvPickingParameters {
    ecvPickingParameters(ecvGenericGLDisplay::PICKING_MODE _mode =
                                 ecvGenericGLDisplay::NO_PICKING,
                         int _centerX = 0,
                         int _centerY = 0,
                         int _pickWidth = 5,
                         int _pickHeight = 5,
                         bool _pickInSceneDB = true,
                         bool _pickInLocalDB = true)
        : mode(_mode),
          centerX(_centerX),
          centerY(_centerY),
          pickWidth(_pickWidth),
          pickHeight(_pickHeight),
          pickInSceneDB(_pickInSceneDB),
          pickInLocalDB(_pickInLocalDB) {}

    ecvGenericGLDisplay::PICKING_MODE mode;
    int centerX;
    int centerY;
    int pickWidth;
    int pickHeight;
    bool pickInSceneDB;
    bool pickInLocalDB;
};

// ============================================================================
// Axes Grid Properties Structure (ParaView-compatible)
// ============================================================================

struct CV_DB_LIB_API AxesGridProperties {
    bool visible = false;
    CCVector3 color = CCVector3(255, 255, 255);
    double lineWidth = 1.0;
    double spacing = 1.0;
    int subdivisions = 10;
    bool showLabels = true;
    double opacity = 1.0;

    bool showGrid = false;
    QString xTitle = "X-Axis";
    QString yTitle = "Y-Axis";
    QString zTitle = "Z-Axis";
    bool xUseCustomLabels = false;
    bool yUseCustomLabels = false;
    bool zUseCustomLabels = false;
    bool useCustomBounds = false;

    QList<QPair<double, QString>> xCustomLabels;
    QList<QPair<double, QString>> yCustomLabels;
    QList<QPair<double, QString>> zCustomLabels;

    double xMin = 0.0, xMax = 1.0;
    double yMin = 0.0, yMax = 1.0;
    double zMin = 0.0, zMax = 1.0;

    AxesGridProperties() = default;
};
