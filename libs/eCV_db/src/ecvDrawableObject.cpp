// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvDrawableObject.h"

#include "ecvDisplayTools.h"

ccDrawableObject::ccDrawableObject() {
    setOpacity(1.0);
    setFixedId(false);
    setRedraw(true);
    setForceRedraw(false);
    setVisible(true);
    setSelected(false);
    lockVisibility(false);
    showColors(false);
    showNormals(false);
    showSF(false);
    enableTempColor(false);
    setTempColor(ecvColor::white, false);
    resetGLTransformation();
    showNameIn3D(false);
}

ccDrawableObject::ccDrawableObject(const ccDrawableObject& object)
    : m_fixedId(object.m_fixedId),
      m_modelRedraw(object.m_modelRedraw),
      m_forceRedraw(object.m_forceRedraw),
      m_opacity(object.m_opacity),
      m_visible(object.m_visible),
      m_selected(object.m_selected),
      m_lockedVisibility(object.m_lockedVisibility),
      m_colorsDisplayed(object.m_colorsDisplayed),
      m_normalsDisplayed(object.m_normalsDisplayed),
      m_sfDisplayed(object.m_sfDisplayed),
      m_tempColor(object.m_tempColor),
      m_colorIsOverridden(object.m_colorIsOverridden),
      m_glTrans(object.m_glTrans),
      m_glTransEnabled(object.m_glTransEnabled),
      m_showNameIn3D(object.m_showNameIn3D) {}

void ccDrawableObject::enableGLTransformation(bool state) {
    m_glTransEnabled = state;
    // if (ecvDisplayTools::GetCurrentScreen())
    //{
    //	ecvDisplayTools::Deprecate3DLayer();
    // }
}

void ccDrawableObject::setGLTransformation(const ccGLMatrix& trans) {
    m_glTrans = trans;
    enableGLTransformation(true);
}

void ccDrawableObject::rotateGL(const ccGLMatrix& rotMat) {
    m_glTrans = rotMat * m_glTrans;
    enableGLTransformation(true);
}

void ccDrawableObject::translateGL(const CCVector3& trans) {
    m_glTrans += trans;
    enableGLTransformation(true);
}

void ccDrawableObject::resetGLTransformation() {
    enableGLTransformation(false);
    m_glTrans.toIdentity();
}

void ccDrawableObject::setTempColor(const ecvColor::Rgb& col,
                                    bool autoActivate /*=true*/) {
    m_tempColor = col;

    if (autoActivate) enableTempColor(true);
}

void ccDrawableObject::getDrawingParameters(glDrawParams& params) const {
    // color override
    if (isColorOverridden()) {
        params.showColors = true;
        params.showNorms = hasNormals() && normalsShown() /*false*/;
        params.showSF = false;
    } else {
        params.showNorms = hasNormals() && normalsShown();
        params.showSF = hasDisplayedScalarField() && sfShown();
        // colors are not displayed if scalar field is displayed
        params.showColors = !params.showSF && hasColors() && colorsShown();
    }
}

bool ccDrawableObject::addClipPlanes(const ccClipPlane& plane) {
    try {
        m_clipPlanes.push_back(plane);
    } catch (const std::bad_alloc&) {
        // not enough memory
        return false;
    }

    return true;
}

void ccDrawableObject::toggleClipPlanes(CC_DRAW_CONTEXT& context, bool enable) {
    if (m_clipPlanes.empty()) {
        return;
    }

    // get the set of OpenGL functions (version 2.1)
    // QOpenGLFunctions_2_1* glFunc =
    // context.glFunctions<QOpenGLFunctions_2_1>(); assert(glFunc != nullptr);

    // if (glFunc == nullptr)
    //	return;

    // GLint maxPlaneCount = 0;
    // glFunc->glGetIntegerv(GL_MAX_CLIP_PLANES, &maxPlaneCount);

    // GLint planeCount = static_cast<GLint>(m_clipPlanes.size());
    // if (planeCount > maxPlaneCount)
    //{
    //	if (enable)
    //	{
    //		CVLog::Warning("[ccDrawableObject::enableClipPlanes] Clipping
    // planes count exceeds the maximum supported number");
    //	}
    //	planeCount = maxPlaneCount;
    // }
    // for (GLint i=0; i < planeCount; ++i)
    //{
    //	GLenum planeIndex = GL_CLIP_PLANE0 + i;
    //	if (enable)
    //	{
    //		glFunc->glClipPlane(planeIndex, m_clipPlanes[i].equation.u);
    //		glFunc->glEnable(planeIndex);
    //	}
    //	else
    //	{
    //		glFunc->glDisable(planeIndex);
    //	}
    // }
}

ccDrawableObject::DisplayState::DisplayState(const ccDrawableObject& dobj)
    : visible(dobj.m_visible),
      colorsDisplayed(dobj.m_colorsDisplayed),
      normalsDisplayed(dobj.m_normalsDisplayed),
      sfDisplayed(dobj.m_sfDisplayed),
      colorIsOverridden(dobj.m_colorIsOverridden),
      showNameIn3D(dobj.m_showNameIn3D) {}

bool ccDrawableObject::pushDisplayState() {
    try {
        m_displayStateStack.emplace_back(new DisplayState(*this));
    } catch (const std::bad_alloc&) {
        CVLog::Warning("Not enough memory to push the current display state");
        return false;
    }

    return true;
}

void ccDrawableObject::popDisplayState(bool apply /*=true*/) {
    if (!m_displayStateStack.empty()) {
        const DisplayState::Shared state = m_displayStateStack.back();
        if (state && apply) {
            applyDisplayState(*state);
        }
        m_displayStateStack.pop_back();
    }
}

void ccDrawableObject::applyDisplayState(const DisplayState& state) {
    if (state.visible != m_visible) setVisible(state.visible);
    if (state.colorsDisplayed != m_colorsDisplayed)
        showColors(state.colorsDisplayed);
    if (state.normalsDisplayed != m_normalsDisplayed)
        showNormals(state.normalsDisplayed);
    if (state.sfDisplayed != m_sfDisplayed) showSF(state.sfDisplayed);
    if (state.colorIsOverridden != m_colorIsOverridden)
        enableTempColor(state.colorIsOverridden);
    if (state.showNameIn3D != m_showNameIn3D) showNameIn3D(state.showNameIn3D);
}
