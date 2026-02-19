// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once



#include <QObject>

#include "CVAppCommon.h"

class QAction;
class QMenu;

class ecvMainAppInterface;
class Mouse3DInput;

class CVAPPCOMMON_LIB_API cc3DMouseManager : public QObject {
    Q_OBJECT

public:
    cc3DMouseManager(ecvMainAppInterface *appInterface, QObject *parent);
    ~cc3DMouseManager();

    //! Gets the menu associated with the 3D mouse
    QMenu *menu() { return m_menu; }

private:
    void enableDevice(bool state, bool silent);
    void releaseDevice();

    void setupMenu();

    void on3DMouseKeyUp(int key);
    void on3DMouseCMDKeyUp(int cmd);
    void on3DMouseKeyDown(int key);
    void on3DMouseCMDKeyDown(int cmd);
    void on3DMouseMove(std::vector<float> &vec);
    void on3DMouseReleased();

    ecvMainAppInterface *m_appInterface;

    Mouse3DInput *m3dMouseInput;

    QMenu *m_menu;
    QAction *m_actionEnable;
};
