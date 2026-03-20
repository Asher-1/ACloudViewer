// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/view/UIShortcuts.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>

namespace sibr {
/*static*/ UIShortcuts& UIShortcuts::global(void) {
    static UIShortcuts instance;
    return instance;
}

void UIShortcuts::list(void) {
    // Sort elements in alphabetical order.
    std::vector<std::pair<std::string, const char*>> elems(_shortcuts.begin(),
                                                           _shortcuts.end());
    std::sort(elems.begin(), elems.end(),
              [](std::pair<std::string, const char*> a,
                 std::pair<std::string, const char*> b) {
                  return b.first < a.first;
              });

    std::ostringstream oss;
    for (auto& pair : elems)
        oss << "  " << std::setw(24) << std::left << pair.first << " : "
            << pair.second << std::endl;
    SIBR_LOG << "List of Shortcuts:\n" << oss.str() << std::endl;
}

void UIShortcuts::add(const std::string& shortcut, const char* desc) {
    std::string lshortcut = shortcut;
    std::transform(lshortcut.begin(), lshortcut.end(), lshortcut.begin(),
                   ::tolower);

    if (_shortcuts.find(lshortcut) == _shortcuts.end())
        _shortcuts[lshortcut] = desc;
    else {
        const char* current = _shortcuts[lshortcut];
        if (current != desc) {
            SIBR_ERR << "conflict with shortcuts.\n"
                        "Trying to register:\n"
                        "["
                     << shortcut << "] : " << desc
                     << "\nBut already exists as:\n"
                        "["
                     << shortcut << "] : " << current << std::endl;
        }
    }
}

}  // namespace sibr
