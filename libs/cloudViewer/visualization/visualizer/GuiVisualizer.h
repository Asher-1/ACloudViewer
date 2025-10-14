// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "visualization/gui/Window.h"

class ccBBox;
class ccHObject;
namespace cloudViewer {

namespace visualization {

namespace gui {
struct Theme;
}

class GuiVisualizer : public gui::Window {
    using Super = gui::Window;

public:
    CLOUDVIEWER_MAKE_ALIGNED_OPERATOR_NEW

    GuiVisualizer(const std::string& title, int width, int height);
    GuiVisualizer(
            const std::vector<std::shared_ptr<const ccHObject>>& geometries,
            const std::string& title,
            int width,
            int height,
            int left,
            int top);
    virtual ~GuiVisualizer();

    void SetTitle(const std::string& title);
    void SetGeometry(std::shared_ptr<const ccHObject> geometry,
                     bool loaded_model);

    bool SetIBL(const char* path);

    /// Loads asynchronously, will return immediately.
    void LoadGeometry(const std::string& path);

    void ExportCurrentImage(const std::string& path);

    void Layout(const gui::LayoutContext& context) override;

    /// Starts the RPC interface. See io/rpc/ReceiverBase for the parameters.
    void StartRPCInterface(const std::string& address, int timeout);

    void StopRPCInterface();

protected:
    // Add custom items to the application menu (only relevant on macOS)
    void AddItemsToAppMenu(
            const std::vector<std::pair<std::string, gui::Menu::ItemId>>&
                    items);

    void OnMenuItemSelected(gui::Menu::ItemId item_id) override;
    void OnDragDropped(const char* path) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void Init();
};

}  // namespace visualization
}  // namespace cloudViewer
