#include "trompeloeil\catch2\trompeloeil.hpp"

#include "ccGenericGLDisplay.h"

class ccGenericGLDisplayMock : public ccGenericGLDisplay {

public:

	MAKE_CONST_MOCK0(getScreenSize, QSize(), override);
	MAKE_MOCK2(redraw, void(bool, bool), override);
	MAKE_MOCK0(toBeRefreshed, void(), override);
	MAKE_MOCK1(refresh, void(bool), override);
	MAKE_MOCK0(invalidateViewport, void(), override);
	MAKE_MOCK0(deprecate3DLayer, void(), override);
	MAKE_CONST_MOCK0(getTextDisplayFont, QFont(), override);
	MAKE_CONST_MOCK0(getLabelDisplayFont, QFont(), override);
	MAKE_MOCK7(displayText, void(QString, int, int, unsigned char, float, const unsigned char*, const QFont*), override);
	MAKE_MOCK4(display3DLabel, void(const QString&, const CCVector3&, const unsigned char*, const QFont&), override);
	MAKE_MOCK1(getGLCameraParameters, void(ccGLCameraParameters&), override);
	MAKE_CONST_MOCK2(toCenteredGLCoordinates, QPointF(int, int), override);
	MAKE_CONST_MOCK2(toCornerGLCoordinates, QPointF(int, int), override);
	MAKE_CONST_MOCK0(getViewportParameters, const ccViewportParameters&(), override);
	MAKE_MOCK5(setupProjectiveViewport, void(const ccGLMatrixd&, float,  float, bool, bool), override);
	MAKE_MOCK0(asWidget, QWidget*(), override);

};
