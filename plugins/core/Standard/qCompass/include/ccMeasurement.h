// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvHObject.h>
#include <ecvMainAppInterface.h>
#include <ecvPointCloud.h>

/*
A template class for all "measurements" made with ccCompass. Contains basic
stuff like highligts, draw colours etc.
*/
class ccMeasurement {
public:
    ccMeasurement() {}

    virtual ~ccMeasurement() {}

    // drawing stuff
    void setDefaultColor(const ecvColor::Rgb& col) { m_normal_colour = col; }
    void setHighlightColor(const ecvColor::Rgb& col) {
        m_highlight_colour = col;
    }
    void setActiveColor(const ecvColor::Rgb& col) { m_active_colour = col; }
    void setAlternateColor(const ecvColor::Rgb& col) {
        m_alternate_colour = col;
    }

    // returns the colour of this measurment object given the active/highlighted
    // state
    ecvColor::Rgb getMeasurementColour() const {
        if (m_isActive) {
            return m_active_colour;
        } else if (m_isAlternate) {
            return m_alternate_colour;
        } else if (m_isHighlighted) {
            return m_highlight_colour;
        }
        return m_normal_colour;
    }

    // set draw state of this measurment
    void setActive(bool isActive) { m_isActive = isActive; }
    void setHighlight(bool isActive) { m_isHighlighted = isActive; }
    void setAlternate(bool isActive) { m_isAlternate = isActive; }
    void setNormal() {
        m_isActive = false;
        m_isHighlighted = false;
        m_isAlternate = false;
    }

protected:
    // drawing variables
    bool m_isActive = false;
    bool m_isHighlighted = false;
    bool m_isAlternate = false;
    ecvColor::Rgb m_active_colour = ecvColor::yellow;
    ecvColor::Rgb m_highlight_colour = ecvColor::green;
    ecvColor::Rgb m_alternate_colour = ecvColor::cyan;
    ecvColor::Rgb m_normal_colour = ecvColor::blue;
};
