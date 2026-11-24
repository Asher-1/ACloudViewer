// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <gmock/gmock.h>

#include "ecvMainAppInterface.h"

class MainAppInterfaceMock : public ecvMainAppInterface {
public:
    MOCK_METHOD(QMainWindow*, getMainWindow, (), (override));
    MOCK_METHOD(void,
                registerOverlayDialog,
                (ccOverlayDialog*, Qt::Corner),
                (override));
    MOCK_METHOD(void, unregisterOverlayDialog, (ccOverlayDialog*), (override));
    MOCK_METHOD(void, updateOverlayDialogsPlacement, (), (override));
    MOCK_METHOD(ccUniqueIDGenerator::Shared,
                getUniqueIDGenerator,
                (),
                (override));
    MOCK_METHOD(ccHObject*, loadFile, (QString, bool), (override));
    MOCK_METHOD(void,
                addToDB,
                (ccHObject*, bool, bool, bool, bool),
                (override));
    MOCK_METHOD(void, removeFromDB, (ccHObject*, bool), (override));
    MOCK_METHOD(ccHObjectContext,
                removeObjectTemporarilyFromDBTree,
                (ccHObject*),
                (override));
    MOCK_METHOD(void,
                putObjectBackIntoDBTree,
                (ccHObject*, const ccHObjectContext&),
                (override));
    MOCK_METHOD(void, setSelectedInDB, (ccHObject*, bool), (override));
    MOCK_METHOD(const ccHObject::Container&,
                getSelectedEntities,
                (),
                (const, override));
    MOCK_METHOD(void,
                dispToConsole,
                (QString, ConsoleMessageLevel),
                (override));
    MOCK_METHOD(void, forceConsoleDisplay, (), (override));
    MOCK_METHOD(ccHObject*, dbRootObject, (), (override));
    MOCK_METHOD(void, redrawAll, (bool, bool), (override));
    MOCK_METHOD(void, refreshAll, (bool, bool), (override));
    MOCK_METHOD(void, enableAll, (), (override));
    MOCK_METHOD(void, disableAll, (), (override));
    MOCK_METHOD(void, refreshSelected, (bool, bool), (override));
    MOCK_METHOD(void, refreshObject, (ccHObject*, bool, bool), (override));
    MOCK_METHOD(void,
                refreshObjects,
                (ccHObject::Container, bool, bool),
                (override));
    MOCK_METHOD(void, resetSelectedBBox, (), (override));
    MOCK_METHOD(void, updateUI, (), (override));
    MOCK_METHOD(void, freezeUI, (bool), (override));
    MOCK_METHOD(ccColorScalesManager*, getColorScalesManager, (), (override));
    MOCK_METHOD(
            void,
            spawnHistogramDialog,
            (const std::vector<unsigned>&, double, double, QString, QString),
            (override));
    MOCK_METHOD(ccPickingHub*, pickingHub, (), (override));
    MOCK_METHOD(void, setView, (CC_VIEW_ORIENTATION), (override));
    MOCK_METHOD(void, toggleActiveWindowCenteredPerspective, (), (override));
    MOCK_METHOD(void, toggleActiveWindowViewerBasedPerspective, (), (override));
    MOCK_METHOD(void, zoomOnSelectedEntities, (), (override));
    MOCK_METHOD(void, zoomOnEntities, (ccHObject*), (override));
    MOCK_METHOD(void, setGlobalZoom, (), (override));
    MOCK_METHOD(void, increasePointSize, (), (override));
    MOCK_METHOD(void, decreasePointSize, (), (override));
    MOCK_METHOD(void, addViewWidget, (QWidget*), (override));
    MOCK_METHOD(QWidget*, getActiveWindow, (), (override));
    MOCK_METHOD(ecvGenericGLDisplay*, getActiveGLDisplay, (), (override));
    MOCK_METHOD(void, toggleExclusiveFullScreen, (bool), (override));
    MOCK_METHOD(void, toggle3DView, (bool), (override));
};
