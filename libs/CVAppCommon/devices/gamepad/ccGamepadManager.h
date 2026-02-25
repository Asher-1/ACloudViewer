// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVAppCommon.h"

// Qt
#include <QObject>

class QAction;
class QMenu;
class QString;

class ecvMainAppInterface;
class GamepadInput;

//! Gamepad manager
class CVAPPCOMMON_LIB_API ccGamepadManager : public QObject {
    Q_OBJECT

public:
    ccGamepadManager(ecvMainAppInterface* appInterface, QObject* parent);
    ~ccGamepadManager();

    //! Returns the menu associated with gamepads
    QMenu* menu() { return m_menu; }

protected:
    void enableDevice(bool state, bool silent, int deviceID = -1);
    void releaseDevice();

    void showMessage(QString message, bool asWarning);
    void setupMenu();
    void setupGamepadInput();

    void onGamepadInput();

private:
    ecvMainAppInterface* m_appInterface;
    GamepadInput* m_gamepadInput;
    QMenu* m_menu;
    QAction* m_actionEnable;
};
