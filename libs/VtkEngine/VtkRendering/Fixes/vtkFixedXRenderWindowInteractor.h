// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <X11/Xlib.h>  // Needed for X types in the public interface
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderingUIModule.h>  // For export macro

class vtkCallbackCommand;

namespace VtkRendering {
class vtkXRenderWindowInteractorInternals;

class vtkXRenderWindowInteractor : public vtkRenderWindowInteractor {
public:
    vtkXRenderWindowInteractor(const vtkXRenderWindowInteractor&) = delete;
    void operator=(const vtkXRenderWindowInteractor&) = delete;

    static vtkXRenderWindowInteractor* New();
    vtkTypeMacro(vtkXRenderWindowInteractor, vtkRenderWindowInteractor);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Initializes the event handlers without an XtAppContext.  This is
     * good for when you don't have a user interface, but you still
     * want to have mouse interaction.
     */
    void Initialize() override;

    /**
     * Break the event loop on 'q','e' keypress. Want more ???
     */
    void TerminateApp() override;

    /**
     * Run the event loop and return. This is provided so that you can
     * implement your own event loop but yet use the vtk event handling as
     * well.
     */
    void ProcessEvents() override;

    ///@{
    /**
     * Enable/Disable interactions.  By default interactors are enabled when
     * initialized.  Initialize() must be called prior to enabling/disabling
     * interaction. These methods are used when a window/widget is being
     * shared by multiple renderers and interactors.  This allows a "modal"
     * display where one interactor is active when its data is to be displayed
     * and all other interactors associated with the widget are disabled
     * when their data is not displayed.
     */
    void Enable() override;
    void Disable() override;
    ///@}

    /**
     * Update the Size data member and set the associated RenderWindow's
     * size.
     */
    void UpdateSize(int, int) override;

    /**
     * Re-defines virtual function to get mouse position by querying X-server.
     */
    void GetMousePosition(int* x, int* y) override;

    void DispatchEvent(XEvent*);

protected:
    vtkXRenderWindowInteractor();
    ~vtkXRenderWindowInteractor() override;

    /**
     * Update the Size data member and set the associated RenderWindow's
     * size but do not resize the XWindow.
     */
    void UpdateSizeNoXResize(int, int);

    // Using static here to avoid destroying context when many apps are open:
    static int NumAppInitialized;

    Display* DisplayId;
    bool OwnDisplay = false;
    Window WindowId;
    Atom KillAtom;
    int PositionBeforeStereo[2];
    vtkXRenderWindowInteractorInternals* Internal;

    // Drag and drop related
    int XdndSourceVersion;
    Window XdndSource;
    Atom XdndFormatAtom;
    Atom XdndURIListAtom;
    Atom XdndTypeListAtom;
    Atom XdndEnterAtom;
    Atom XdndPositionAtom;
    Atom XdndDropAtom;
    Atom XdndActionCopyAtom;
    Atom XdndStatusAtom;
    Atom XdndFinishedAtom;

    ///@{
    /**
     * X-specific internal timer methods. See the superclass for detailed
     * documentation.
     */
    int InternalCreateTimer(int timerId,
                            int timerType,
                            unsigned long duration) override;
    int InternalDestroyTimer(int platformTimerId) override;
    ///@}

    void FireTimers();

    /**
     * This will start up the X event loop and never return. If you
     * call this method it will loop processing X events until the
     * application is exited.
     */
    void StartEventLoop() override;

    /**
     * Deallocate X resource that may have been allocated
     * Also calls finalize on the render window if available
     */
    void Finalize();
};
}  // namespace VtkRendering
