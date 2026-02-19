// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Qt
#include <QString>

// CV_DB_LIB
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>

class QMainWindow;
class QWidget;
class ccColorScalesManager;
class ccOverlayDialog;
class ccPickingHub;

/**
 * @class ecvMainAppInterface
 * @brief Main application interface for plugins
 * 
 * Abstract interface that provides plugins with access to the main
 * CloudViewer application functionality. Plugins receive an instance
 * of this interface and can use it to:
 * 
 * - Access the main window and UI elements
 * - Manage overlay dialogs in the MDI area
 * - Load files and add/remove entities from the database
 * - Control display and rendering (refresh, redraw, zoom)
 * - Manage entity selection
 * - Display console messages
 * - Access shared resources (color scales, picking hub, etc.)
 * 
 * This interface follows the facade pattern, providing a simplified
 * and stable API for plugin development while insulating plugins
 * from internal application implementation details.
 * 
 * @see ccPluginInterface
 */
class ecvMainAppInterface {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~ecvMainAppInterface() = default;

    /**
     * @brief Get main application window
     * @return Pointer to main window
     */
    virtual QMainWindow* getMainWindow() = 0;

    /**
     * @brief Register an overlay dialog in MDI area
     * 
     * Overlay dialogs are displayed above the 3D views in the central
     * MDI area. Position is relative to one of the MDI area's corners
     * and automatically updates when the window is resized.
     * 
     * Registered dialogs are automatically released on application exit.
     * 
     * @param dlg Overlay dialog to register
     * @param pos Corner position for dialog placement
     * 
     * @note Call updateOverlayDialogsPlacement() after registration
     * @note Consider freezing UI during tool operation to prevent
     *       other overlay dialogs from appearing
     */
    virtual void registerOverlayDialog(ccOverlayDialog* dlg,
                                       Qt::Corner pos) = 0;

    /**
     * @brief Unregister and delete an overlay dialog
     * 
     * @param dlg Overlay dialog to unregister
     * @warning Dialog object will be deleted (via QObject::deleteLater)
     */
    virtual void unregisterOverlayDialog(ccOverlayDialog* dlg) = 0;

    /**
     * @brief Force update of all overlay dialog positions
     * 
     * Recalculates and updates positions for all registered overlay dialogs.
     * Call after window resize or dialog registration.
     */
    virtual void updateOverlayDialogsPlacement() = 0;

    /**
     * @brief Get unique ID generator
     * @return Shared pointer to ID generator
     */
    virtual ccUniqueIDGenerator::Shared getUniqueIDGenerator() = 0;

    /**
     * @brief Load a file
     * 
     * Attempts to load entities from the specified file using
     * appropriate file filters.
     * 
     * @param filename Path to file to load
     * @param silent Suppress error/info dialogs
     * @return Loaded entity (or nullptr on failure)
     */
    virtual ccHObject* loadFile(QString filename, bool silent) = 0;

    /**
     * @brief Add entity to database
     * 
     * Adds entity to the database tree and optionally performs
     * additional operations (zoom, expand tree, etc.).
     * 
     * @param obj Entity to add
     * @param updateZoom Update zoom to fit new entity
     * @param autoExpandDBTree Auto-expand DB tree to show new entity
     * @param checkDimensions Check entity dimensions and warn if large
     * @param autoRedraw Automatically redraw displays
     */
    virtual void addToDB(ccHObject* obj,
                         bool updateZoom = false,
                         bool autoExpandDBTree = true,
                         bool checkDimensions = false,
                         bool autoRedraw = true) = 0;

    /**
     * @brief Remove entity from database tree
     * 
     * Removes entity and automatically detaches it from parent.
     * @param obj Entity to remove
     * @param autoDelete Automatically delete object after removal
     */
    virtual void removeFromDB(ccHObject* obj, bool autoDelete = true) = 0;

    /**
     * @struct ccHObjectContext
     * @brief Backup context for temporarily removed objects
     * 
     * Stores parent-child relationship and flags for objects
     * temporarily removed from the database tree. Used to restore
     * objects to their original state.
     * 
     * @see removeObjectTemporarilyFromDBTree()
     * @see putObjectBackIntoDBTree()
     */
    struct ccHObjectContext {
        ccHObject* parent = nullptr;  ///< Parent object
        int childFlags = 0;           ///< Child dependency flags
        int parentFlags = 0;          ///< Parent dependency flags
    };

    /**
     * @brief Temporarily remove object from database tree
     * 
     * Removes object from tree while preserving context for later
     * restoration. Use before making structural modifications to
     * the database tree.
     * 
     * @param obj Object to remove temporarily
     * @return Context information for restoration
     * @warning May change currently selected entities
     * @see putObjectBackIntoDBTree()
     */
    virtual ccHObjectContext removeObjectTemporarilyFromDBTree(
            ccHObject* obj) = 0;

    /**
     * @brief Restore object to database tree
     * 
     * Adds object back to tree using saved context information.
     * Call after completing database tree modifications.
     * 
     * @param obj Object to restore
     * @param context Saved context from removeObjectTemporarilyFromDBTree()
     * @see removeObjectTemporarilyFromDBTree()
     */
    virtual void putObjectBackIntoDBTree(ccHObject* obj,
                                         const ccHObjectContext& context) = 0;

    /**
     * @brief Select or deselect entity in database tree
     * @param obj Entity to modify
     * @param selected New selection state (true = selected)
     */
    virtual void setSelectedInDB(ccHObject* obj, bool selected) = 0;

    /**
     * @brief Get currently selected entities
     * @return Const reference to container of selected entities
     */
    virtual const ccHObject::Container& getSelectedEntities() const = 0;

    /**
     * @brief Check if any entities are selected
     * @return true if at least one entity is selected
     */
    bool haveSelection() const { return !getSelectedEntities().empty(); }

    /**
     * @brief Check if exactly one entity is selected
     * @return true if selection contains exactly one entity
     */
    bool haveOneSelection() const { return getSelectedEntities().size() == 1; }

    /**
     * @brief Console message severity level
     */
    enum ConsoleMessageLevel {
        STD_CONSOLE_MESSAGE = 0,  ///< Standard information message
        WRN_CONSOLE_MESSAGE = 1,  ///< Warning message
        ERR_CONSOLE_MESSAGE = 2,  ///< Error message
    };

    /**
     * @brief Display message in console
     * 
     * Prints message to application console with specified severity level.
     * Message formatting/coloring depends on level.
     * 
     * @param message Message text to display
     * @param level Message severity level (default: standard)
     */
    virtual void dispToConsole(
            QString message,
            ConsoleMessageLevel level = STD_CONSOLE_MESSAGE) = 0;

    /**
     * @brief Force console widget to be displayed
     * 
     * Makes console window visible if hidden.
     */
    virtual void forceConsoleDisplay() = 0;

    /**
     * @brief Get database root object
     * @return Pointer to root of entity database tree
     */
    virtual ccHObject* dbRootObject() = 0;

    /**
     * @brief Refresh all displays with pending updates
     * 
     * Redraws all GL windows that have the refresh flag set.
     * @param only2D Refresh only 2D layer (false = full 3D+2D refresh)
     * @param forceRedraw Force complete redraw (default: true)
     * @see ccDrawableObject::prepareDisplayForRefresh
     */
    virtual void refreshAll(bool only2D = false, bool forceRedraw = true) = 0;
    
    /**
     * @brief Redraw all displays (alias for refreshAll)
     * @param only2D Redraw only 2D layer
     * @param forceRedraw Force complete redraw
     */
    virtual void redrawAll(bool only2D = false, bool forceRedraw = true) {
        refreshAll(only2D, forceRedraw);
    };

    /**
     * @brief Enable all display windows
     */
    virtual void enableAll() = 0;

    /**
     * @brief Disable all display windows
     */
    virtual void disableAll() = 0;

    /**
     * @brief Refresh selected entities
     * @param only2D Refresh only 2D layer
     * @param forceRedraw Force complete redraw
     */
    virtual void refreshSelected(bool only2D = false,
                                 bool forceRedraw = true) = 0;
    
    /**
     * @brief Refresh specific object
     * @param obj Object to refresh
     * @param only2D Refresh only 2D layer
     * @param forceRedraw Force complete redraw
     */
    virtual void refreshObject(ccHObject* obj,
                               bool only2D = false,
                               bool forceRedraw = true) = 0;
    
    /**
     * @brief Refresh multiple objects
     * @param objs Objects to refresh
     * @param only2D Refresh only 2D layer
     * @param forceRedraw Force complete redraw
     */
    virtual void refreshObjects(ccHObject::Container objs,
                                bool only2D = false,
                                bool forceRedraw = true) = 0;
    
    /**
     * @brief Reset bounding box for selected entities
     */
    virtual void resetSelectedBBox() = 0;

    /**
     * @brief Update UI to reflect selection state
     * 
     * Updates menus and property browser to match current entity selection.
     * Should be called after any changes to selected entities.
     */
    virtual void updateUI() = 0;

    /**
     * @brief Freeze or unfreeze user interface
     * 
     * Disables/enables UI elements to prevent user interaction
     * during lengthy operations.
     * @param state true = freeze, false = unfreeze
     */
    virtual void freezeUI(bool state) = 0;

    /**
     * @brief Get color scales manager
     * @return Pointer to singleton color scales manager
     */
    virtual ccColorScalesManager* getColorScalesManager() = 0;

    /**
     * @brief Display histogram dialog
     * 
     * Shows a histogram visualization dialog with specified data.
     * @param histoValues Histogram bin values
     * @param minVal Minimum value (X-axis)
     * @param maxVal Maximum value (X-axis)
     * @param title Dialog/histogram title
     * @param xAxisLabel X-axis label text
     */
    virtual void spawnHistogramDialog(const std::vector<unsigned>& histoValues,
                                      double minVal,
                                      double maxVal,
                                      QString title,
                                      QString xAxisLabel) = 0;

    /**
     * @brief Get picking hub
     * @return Pointer to picking hub (or nullptr if unavailable)
     */
    virtual ccPickingHub* pickingHub() { return nullptr; }

    /**
     * @brief Set standard view orientation
     * @param view View orientation (top, front, side, etc.)
     */
    virtual void setView(CC_VIEW_ORIENTATION view) = 0;
    
    /**
     * @brief Toggle object-centered perspective in active window
     */
    virtual void toggleActiveWindowCenteredPerspective() = 0;
    
    /**
     * @brief Toggle viewer-centered perspective in active window
     */
    virtual void toggleActiveWindowViewerBasedPerspective() = 0;

    /**
     * @brief Zoom to fit selected entities
     */
    virtual void zoomOnSelectedEntities() = 0;
    
    /**
     * @brief Zoom to fit specific entities
     * @param obj Entities to zoom to
     */
    virtual void zoomOnEntities(ccHObject* obj) = 0;
    
    /**
     * @brief Set global zoom (fit all entities)
     */
    virtual void setGlobalZoom() = 0;

    /**
     * @brief Increase point display size
     */
    virtual void increasePointSize() = 0;
    
    /**
     * @brief Decrease point display size
     */
    virtual void decreasePointSize() = 0;

    /**
     * @brief Add widget to MDI area
     * @param viewWidget Widget to add
     */
    virtual void addWidgetToQMdiArea(QWidget* viewWidget) = 0;
    
    /**
     * @brief Get currently active display window
     * @return Pointer to active window widget
     */
    virtual QWidget* getActiveWindow() = 0;

    /**
     * @brief Toggle exclusive fullscreen mode
     * @param state Fullscreen state (true = enable)
     */
    virtual void toggleExclusiveFullScreen(bool state) = 0;
    
    /**
     * @brief Toggle 3D view visibility
     * @param state View visibility (true = show)
     */
    virtual void toggle3DView(bool state) = 0;
};
