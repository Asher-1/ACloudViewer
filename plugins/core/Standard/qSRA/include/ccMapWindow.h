// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// ECV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvScalarField.h>

//! 2D map display window
class ccMapWindow : public QWidget {
public:
    //! Default constructor
    explicit ccMapWindow(QWidget* parent = 0)
        : QWidget(parent), m_sfForRampDisplay(0), m_showSF(true) {}

    //! Destructor
    virtual ~ccMapWindow() { setAssociatedScalarField(0); }

    //! Sets associated scalar-field
    /** This scalar field will be used for color ramp display.
     **/
    void setAssociatedScalarField(ccScalarField* sf) {
        if (m_sfForRampDisplay != sf) {
            if (m_sfForRampDisplay) m_sfForRampDisplay->release();

            m_sfForRampDisplay = sf;

            if (m_sfForRampDisplay) m_sfForRampDisplay->link();
        }
    }

    //! Whether to show associated SF or not
    void showSF(bool state) { m_showSF = state; }

    //! Returns whether associated SF should be shown or not
    bool sfShown() const { return m_showSF; }

    //! Returns associated scalar field
    ccScalarField* getAssociatedScalarField() const {
        return m_sfForRampDisplay;
    }

    // inherited fro ccGLWindow
    virtual void getContext(CC_DRAW_CONTEXT& context) {
        // ccGLWindow::getContext(context);
        ecvDisplayTools::GetContext(context);

        if (m_showSF) {
            // override sf that will be used for color ramp display
            context.sfColorScaleToDisplay = m_sfForRampDisplay;
        }
    }

protected:
    //! Associated scalar field
    ccScalarField* m_sfForRampDisplay;

    //! Whether to show or not the associated SF
    bool m_showSF;
};
