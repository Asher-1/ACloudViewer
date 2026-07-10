// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "ecvDrawContext.h"
#include "ecvGLMatrix.h"

// STL
#include <algorithm>
#include <vector>

// CV_CORE_LIB
#include <CVGeom.h>

class ecvGenericGLDisplay;

//! Simple (clipping) plane equation
struct ccClipPlane {
    Tuple4Tpl<double> equation;
};
using ccClipPlaneSet = std::vector<ccClipPlane>;

//! Generic interface for (3D) drawable entities
class CV_DB_LIB_API ccDrawableObject {
public:
    //! Default constructor
    ccDrawableObject();
    //! Copy constructor
    ccDrawableObject(const ccDrawableObject& object);

    virtual ~ccDrawableObject() = default;

public:  // display association (multi-window support)
    /// Returns the primary associated GL display (first in the binding set),
    /// or nullptr if unbound. Backward-compatible single-display accessor.
    virtual ecvGenericGLDisplay* getDisplay() const;

    /// Associates this entity with a SINGLE display (replaces all previous
    /// bindings). Equivalent to clearDisplays() + addToDisplay(display).
    /// Pass nullptr to explicitly unbind from all displays (display=None).
    virtual void setDisplay(ecvGenericGLDisplay* display);

    /// Adds a display to this entity's binding set.
    virtual void addToDisplay(ecvGenericGLDisplay* display);

    /// Removes a specific display from the binding set.
    virtual void removeFromDisplay(const ecvGenericGLDisplay* display);

    /// Clears all display bindings (entity shows nowhere = None).
    virtual void clearAllDisplays();

    /// Resets to legacy mode: entity shows in ALL windows (no explicit
    /// binding).
    virtual void setDisplayAll();

    /// Returns the full set of bound displays.
    const std::vector<ecvGenericGLDisplay*>& getDisplays() const {
        return m_displays;
    }

    /// Returns true if entity has explicit display binding (vs legacy "all").
    bool hasExplicitDisplayBinding() const { return m_displayBindingExplicit; }

public:  // drawing and drawing options
    //! Draws entity and its children
    virtual void draw(CC_DRAW_CONTEXT& context) = 0;

    //! Returns whether entity is visible or not
    inline virtual bool isVisible() const { return m_visible; }
    //! Sets entity visibility
    inline virtual void setVisible(bool state) {
        if (state && !m_visible) setRedraw(true);
        m_visible = state;
    }

    //! Toggles visibility
    inline virtual void toggleVisibility() { setVisible(!isVisible()); }

    //! Returns whether entity is to be redraw
    inline virtual bool isRedraw() const { return m_modelRedraw; }
    //! Sets entity redraw mode
    inline virtual void setRedraw(bool state) { m_modelRedraw = state; }
    //! Sets force redraw
    inline virtual void setForceRedraw(bool state) { m_forceRedraw = state; }

    inline virtual void setFixedId(bool state) { m_fixedId = state; }
    inline virtual bool isFixedId() { return m_fixedId; }

    //! Returns whether visibility is locked or not
    inline virtual bool isVisibilityLocked() const {
        return m_lockedVisibility;
    }
    //! Locks/unlocks visibility
    /** If visibility is locked, the user won't be able to modify it
            (via the properties tree for instance).
    **/
    inline virtual void lockVisibility(bool state) {
        m_lockedVisibility = state;
    }

    //! Returns whether entity is selected or not
    inline virtual bool isSelected() const { return m_selected; }
    //! Selects/Unselects entity
    inline virtual void setSelected(bool state) { m_selected = state; }

    //! Returns main OpenGL parameters for this entity
    /** These parameters are deduced from the visibility states
            of its different features (points, normals, etc.).
            \param params a glDrawParams structure
    **/
    virtual void getDrawingParameters(glDrawParams& params) const;

    //! Returns whether colors are enabled or not
    inline virtual bool hasColors() const { return false; }
    //! Returns whether colors are shown or not
    inline virtual bool colorsShown() const { return m_colorsDisplayed; }
    //! Sets colors visibility
    inline virtual void showColors(bool state) { m_colorsDisplayed = state; }
    //! Toggles colors display state
    inline virtual void toggleColors() { showColors(!colorsShown()); }

    //! Returns whether normals are enabled or not
    inline virtual bool hasNormals() const { return false; }
    //! Returns whether normals are shown or not
    inline virtual bool normalsShown() const { return m_normalsDisplayed; }
    //! Sets normals visibility
    inline virtual void showNormals(bool state) { m_normalsDisplayed = state; }
    //! Toggles normals display state
    inline virtual void toggleNormals() { showNormals(!normalsShown()); }

public:  // scalar fields
    //! Returns whether an active scalar field is available or not
    inline virtual bool hasDisplayedScalarField() const { return false; }

    //! Returns whether one or more scalar fields are instantiated
    /** WARNING: doesn't mean a scalar field is currently displayed
            (see ccDrawableObject::hasDisplayedScalarField).
    **/
    inline virtual bool hasScalarFields() const { return false; }

    //! Sets active scalarfield visibility
    inline virtual void showSF(bool state) {
        m_sfDisplayed = state;
        setRedraw(state);
    }

    //! Toggles SF display state
    inline virtual void toggleSF() { showSF(!sfShown()); }

    //! Returns whether active scalar field is visible
    inline virtual bool sfShown() const { return m_sfDisplayed; }

public:  //(Mesh) materials
    //! Toggles material display state
    virtual void toggleMaterials() {}  // does nothing by default!

public:  // Name display in 3D
    //! Sets whether name should be displayed in 3D
    inline virtual void showNameIn3D(bool state) { m_showNameIn3D = state; }

    //! Returns whether name is displayed in 3D or not
    inline virtual bool nameShownIn3D() const { return m_showNameIn3D; }

    //! Toggles name in 3D display state
    inline virtual void toggleShowName() { showNameIn3D(!nameShownIn3D()); }

public:  // Temporary color
    //! Returns whether colors are currently overridden by a temporary (unique)
    //! color
    /** See ccDrawableObject::setTempColor.
     **/
    inline virtual bool isColorOverridden() const {
        return m_colorIsOverridden;
    }

    //! Returns current temporary (unique) color
    inline virtual const ecvColor::Rgb& getTempColor() const {
        return m_tempColor;
    }

    //! Sets current temporary (unique)
    /** \param col rgb color
            \param autoActivate auto activates temporary color
    **/
    virtual void setTempColor(const ecvColor::Rgb& col,
                              bool autoActivate = true);

    //! Set temporary color activation state
    inline virtual void enableTempColor(bool state) {
        m_colorIsOverridden = state;
    }

    // Get opacity
    inline virtual float getOpacity() const { return m_opacity; }

    //! Set opacity activation state
    inline virtual void setOpacity(float opacity) {
        m_opacity = opacity;
        setRedraw(false);
    }

    //! Point Gaussian shader presets (matching ParaView)
    enum PointGaussianShaderPreset {
        PG_GAUSSIAN_BLUR = 0,
        PG_SPHERE,
        PG_BLACK_EDGED_CIRCLE,
        PG_PLAIN_CIRCLE,
        PG_TRIANGLE,
        PG_SQUARE_OUTLINE,
        PG_PRESET_COUNT
    };

    //! Returns whether Point Gaussian (splat) rendering is enabled
    inline bool pointGaussianEnabled() const { return m_pointGaussianEnabled; }

    //! Enables/disables Point Gaussian (splat) rendering
    inline void setPointGaussianEnabled(bool state) {
        m_pointGaussianEnabled = state;
        setRedraw(false);
    }

    //! Returns the Point Gaussian splat radius
    inline double pointGaussianRadius() const { return m_pointGaussianRadius; }

    //! Sets the Point Gaussian splat radius
    inline void setPointGaussianRadius(double radius) {
        m_pointGaussianRadius = radius;
        setRedraw(false);
    }

    //! Returns the Point Gaussian shader preset
    inline int pointGaussianShaderPreset() const {
        return m_pointGaussianShaderPreset;
    }

    //! Sets the Point Gaussian shader preset
    inline void setPointGaussianShaderPreset(int preset) {
        m_pointGaussianShaderPreset = preset;
        setRedraw(false);
    }

    //! Returns whether Point Gaussian emissive mode is enabled
    inline bool pointGaussianEmissive() const {
        return m_pointGaussianEmissive;
    }

    //! Enables/disables Point Gaussian emissive mode
    inline void setPointGaussianEmissive(bool state) {
        m_pointGaussianEmissive = state;
        setRedraw(false);
    }

public:  // Transformation matrix management (for display only)
    //! Associates entity with a GL transformation (rotation + translation)
    /** \warning FOR DISPLAY PURPOSE ONLY (i.e. should only be temporary)
            If the associated GL transformation is enabled (see
            ccDrawableObject::enableGLTransformation), it will
            be applied before displaying this entity.
            However it will not be taken into account by any cloudViewer
    algorithm (distance computation, etc.) for instance. Note: GL transformation
    is automatically enabled.
    **/
    virtual void setGLTransformation(const ccGLMatrix& trans);

    //! Enables/disables associated GL transformation
    /** See ccDrawableObject::setGLTransformation.
     **/
    virtual void enableGLTransformation(bool state);

    //! Returns whether a GL transformation is enabled or not
    inline virtual bool isGLTransEnabled() const { return m_glTransEnabled; }

    //! Returns associated GL transformation
    /** See ccDrawableObject::setGLTransformation.
     **/
    inline virtual const ccGLMatrix& getGLTransformation() const {
        return m_glTrans;
    }

    //! Resets associated GL transformation
    /** GL transformation is reset to identity.
            Note: GL transformation is automatically disabled.
            See ccDrawableObject::setGLTransformation.
    **/
    virtual void resetGLTransformation();

    //! Multiplies (left) current GL transformation by a rotation matrix
    /** 'GLtrans = M * GLtrans'
            Note: GL transformation is automatically enabled.
            See ccDrawableObject::setGLTransformation.
    **/
    virtual void rotateGL(const ccGLMatrix& rotMat);

    //! Translates current GL transformation by a rotation matrix
    /** 'GLtrans = GLtrans + T'
            Note: GL transformation is automatically enabled.
            See ccDrawableObject::setGLTransformation.
    **/
    virtual void translateGL(const CCVector3& trans);

public:  // clipping planes
    //! Removes all clipping planes (if any)
    virtual void removeAllClipPlanes() { m_clipPlanes.resize(0); }

    //! Registers a new clipping plane
    /** \return false if the planes couldn't be added (not enough memory)
     **/
    virtual bool addClipPlanes(const ccClipPlane& plane);

    //! Enables or disables clipping planes (OpenGL)
    /** \warning If enabling the clipping planes, be sure to call this method
     *AFTER the model view matrix has been set.
     **/
    virtual void toggleClipPlanes(CC_DRAW_CONTEXT& context, bool enable);

public:  // push and pop display state
    //! Display state
    struct DisplayState {
        DisplayState() {}
        DisplayState(const ccDrawableObject& dobj);

        using Shared = QSharedPointer<DisplayState>;

        bool visible = false;
        bool colorsDisplayed = false;
        bool normalsDisplayed = false;
        bool sfDisplayed = false;
        bool colorIsOverridden = false;
        bool showNameIn3D = false;
    };

    //! Pushes the current display state
    virtual bool pushDisplayState();

    //! Pops the last pushed display state
    virtual void popDisplayState(bool apply = true);

    //! Applies a display state
    virtual void applyDisplayState(const DisplayState& state);

protected:  // members
    /// Bound displays for this entity (multi-window support).
    /// Empty + m_displayBindingExplicit=false → legacy "show in all windows"
    /// Empty + m_displayBindingExplicit=true  → display=None (show nowhere)
    /// Non-empty → show only in these specific windows
    std::vector<ecvGenericGLDisplay*> m_displays;

    /// Whether the display binding is explicit (user-controlled).
    /// false = legacy mode: entity shows in all windows (default for new
    /// entities). true  = explicit: entity shows only in m_displays (or nowhere
    /// if empty).
    bool m_displayBindingExplicit = false;

    bool m_fixedId;
    bool m_modelRedraw;
    bool m_forceRedraw;
    float m_opacity;

    //! Specifies whether the object is visible or not
    /** Note: this does not influence the children visibility
     **/
    bool m_visible;

    //! Specifies whether the object is selected or not
    bool m_selected;

    //! Specifies whether the visibility can be changed by user or not
    bool m_lockedVisibility;

    //! Specifies whether colors should be displayed
    bool m_colorsDisplayed;
    //! Specifies whether normals should be displayed
    bool m_normalsDisplayed;
    //! Specifies whether scalar field should be displayed
    bool m_sfDisplayed;

    //! Temporary (unique) color
    ecvColor::Rgb m_tempColor;
    //! Temporary (unique) color activation state
    bool m_colorIsOverridden;

    //! Current GL transformation
    /** See ccDrawableObject::setGLTransformation.
     **/
    ccGLMatrix m_glTrans;
    //! Current GL transformation activation state
    /** See ccDrawableObject::setGLTransformation.
     **/
    bool m_glTransEnabled;

    //! Whether name is displayed in 3D or not
    bool m_showNameIn3D;
    //! Last 2D position of the '3D' name
    CCVector3d m_nameIn3DPos;

    //! Active clipping planes (used for display only)
    ccClipPlaneSet m_clipPlanes;

    //! Point Gaussian (splat) rendering enabled
    bool m_pointGaussianEnabled;

    //! Point Gaussian splat radius
    double m_pointGaussianRadius;

    //! Point Gaussian shader preset index
    int m_pointGaussianShaderPreset;

    //! Point Gaussian emissive mode
    bool m_pointGaussianEmissive;

    //! The stack of pushed display states
    std::vector<DisplayState::Shared> m_displayStateStack;
};
