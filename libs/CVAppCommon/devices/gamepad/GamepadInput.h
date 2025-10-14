// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
// ##########################################################################
// #                                                                        #
// #                              CLOUDCOMPARE                              #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 or later of the License.      #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                  COPYRIGHT: Daniel Girardeau-Montaut                   #
// #                                                                        #
// ##########################################################################

#include "CVAppCommon.h"

// cloudViewer
#include <CVConst.h>

// qCC_db
#include <ecvGLMatrix.h>

// Qt
#include <QGamepad>
#include <QTimer>

class QMainWindow;

//! Gaempad handler
class CVAPPCOMMON_LIB_API GamepadInput : public QGamepad {
    Q_OBJECT

public:
    //! Default constructor
    explicit GamepadInput(QObject* parent = nullptr);
    //! Destructor
    virtual ~GamepadInput();

    void start();
    void stop();

    //! Updates a window with the current gamepad state
    void update(QMainWindow* win);

Q_SIGNALS:

    void updated();

protected:
    void updateInternalState();

private:
    //! Timer to poll the gamepad state
    QTimer m_timer;

    //! Last state
    CCVector3 m_panning;
    bool m_hasPanning;
    CCVector3 m_translation;
    bool m_hasTranslation;
    ccGLMatrixd m_rotation;
    bool m_hasRotation;
    float m_zoom;
};
