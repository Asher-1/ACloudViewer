#include "trompeloeil\catch2\trompeloeil.hpp"

#include "ccGLWindow.h"
#include "ccMainAppInterface.h"

class MainAppInterfaceMock : public ccMainAppInterface {

public:

	MAKE_MOCK0(getMainWindow, QMainWindow*(), override);
	MAKE_MOCK0(getActiveGLWindow, ccGLWindow*(), override);
	MAKE_CONST_MOCK2(createGLWindow, void(ccGLWindow*&, QWidget*&), override);
	MAKE_CONST_MOCK1(destroyGLWindow, void(ccGLWindow*), override);
	MAKE_MOCK2(registerOverlayDialog, void(ccOverlayDialog*, Qt::Corner), override);
	MAKE_MOCK1(unregisterOverlayDialog, void(ccOverlayDialog*), override);
	MAKE_MOCK0(updateOverlayDialogsPlacement, void(), override);
	MAKE_MOCK0(getUniqueIDGenerator, ccUniqueIDGenerator::Shared(), override);
	MAKE_MOCK5(addToDB, void(ccHObject*, bool, bool, bool, bool), override);
	MAKE_MOCK2(removeFromDB, void(ccHObject*, bool), override);
	MAKE_MOCK1(removeObjectTemporarilyFromDBTree, ccHObjectContext(ccHObject*), override);
	MAKE_MOCK2(putObjectBackIntoDBTree, void(ccHObject*, const ccHObjectContext&), override);
	MAKE_MOCK2(setSelectedInDB, void(ccHObject*, bool), override);
	MAKE_CONST_MOCK0(getSelectedEntities, const ccHObject::Container&(), override);
	MAKE_MOCK2(dispToConsole, void(QString, ConsoleMessageLevel), override);
	MAKE_MOCK0(forceConsoleDisplay, void(), override);
	MAKE_MOCK0(dbRootObject, ccHObject*(), override);
	MAKE_MOCK1(redrawAll, void(bool), override);
	MAKE_MOCK1(refreshAll, void(bool), override);
	MAKE_MOCK0(enableAll, void(), override);
	MAKE_MOCK0(disableAll, void(), override);
	MAKE_MOCK1(disableAllBut, void(ccGLWindow*), override);
	MAKE_MOCK0(updateUI, void(), override);
	MAKE_MOCK1(freezeUI, void(bool), override);
	MAKE_MOCK0(getColorScalesManager, ccColorScalesManager*(), override);
	MAKE_MOCK5(spawnHistogramDialog, void(const std::vector<unsigned>&, double, double, QString, QString), override);
	MAKE_MOCK0(pickingHub, ccPickingHub*(), override);
	MAKE_MOCK1(setView, void(CC_VIEW_ORIENTATION), override);
	MAKE_MOCK0(toggleActiveWindowCenteredPerspective, void(), override);
	MAKE_MOCK0(toggleActiveWindowCustomLight, void(), override);
	MAKE_MOCK0(toggleActiveWindowSunLight, void(), override);
	MAKE_MOCK0(toggleActiveWindowViewerBasedPerspective, void(), override);
	MAKE_MOCK0(zoomOnSelectedEntities, void(), override);
	MAKE_MOCK0(setGlobalZoom, void(), override);
	MAKE_MOCK0(increasePointSize, void(), override);
	MAKE_MOCK0(decreasePointSize, void(), override);

};
