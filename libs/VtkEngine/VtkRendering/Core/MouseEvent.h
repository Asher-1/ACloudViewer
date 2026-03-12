// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file MouseEvent.h
 *  @brief Mouse event struct for ImageVis and ImageViewer callback handling
 */

namespace VtkRendering {

/** @struct MouseEvent
 *  @brief Simple event struct replacing pcl::visualization::MouseEvent.
 *  Used by ImageVis and ImageViewer for mouse callback handling.
 */
struct MouseEvent {
    enum Type {
        MouseMove,
        MouseButtonPress,
        MouseButtonRelease,
        MouseScrollDown,
        MouseScrollUp,
        MouseDblClick
    };

    enum MouseButton {
        NoButton,
        LeftButton,
        MiddleButton,
        RightButton,
        VScroll
    };

    Type type;
    MouseButton button;
    int x;
    int y;
    bool alt;
    bool ctrl;
    bool shift;

    /// @param t Event type
    /// @param b Mouse button
    /// @param x_pos X coordinate
    /// @param y_pos Y coordinate
    /// @param alt_key Alt modifier
    /// @param ctrl_key Ctrl modifier
    /// @param shift_key Shift modifier
    MouseEvent(Type t = MouseMove,
               MouseButton b = NoButton,
               int x_pos = 0,
               int y_pos = 0,
               bool alt_key = false,
               bool ctrl_key = false,
               bool shift_key = false)
        : type(t),
          button(b),
          x(x_pos),
          y(y_pos),
          alt(alt_key),
          ctrl(ctrl_key),
          shift(shift_key) {}

    int getX() const { return x; }
    int getY() const { return y; }
    Type getType() const { return type; }
    MouseButton getButton() const { return button; }
    void setType(Type t) { type = t; }
    void setButton(MouseButton b) { button = b; }
};

}  // namespace VtkRendering
