// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file KeyboardEvent.h
 *  @brief Keyboard event struct and modifier enum (replaces
 * pcl::visualization::KeyboardEvent)
 */

#include <string>

namespace VtkRendering {

/// Keyboard modifier enum replacing
/// pcl::visualization::InteractorKeyboardModifier.
enum InteractorKeyboardModifier {
    INTERACTOR_KB_MOD_ALT,
    INTERACTOR_KB_MOD_CTRL,
    INTERACTOR_KB_MOD_SHIFT
};

/** @struct KeyboardEvent
 *  @brief Simple event struct replacing pcl::visualization::KeyboardEvent.
 */
struct KeyboardEvent {
    bool key_down;
    std::string key_sym;
    int key_code;
    bool alt;
    bool ctrl;
    bool shift;

    /// @param down true if key pressed
    /// @param sym Key symbol string
    /// @param code Key code
    /// @param alt_key Alt modifier
    /// @param ctrl_key Ctrl modifier
    /// @param shift_key Shift modifier
    KeyboardEvent(bool down = false,
                  const std::string& sym = "",
                  int code = 0,
                  bool alt_key = false,
                  bool ctrl_key = false,
                  bool shift_key = false)
        : key_down(down),
          key_sym(sym),
          key_code(code),
          alt(alt_key),
          ctrl(ctrl_key),
          shift(shift_key) {}

    bool isAltPressed() const { return alt; }
    bool isCtrlPressed() const { return ctrl; }
    bool isShiftPressed() const { return shift; }
    int getKeyCode() const { return key_code; }
    const std::string& getKeySym() const { return key_sym; }
    bool keyDown() const { return key_down; }
    bool keyUp() const { return !key_down; }
};

}  // namespace VtkRendering
