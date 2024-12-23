// ----------------------------------------------------------------------------
// -                        CloudViewer: Asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 Asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#ifdef __APPLE__

#import <Cocoa/Cocoa.h>
#import <CoreServices/CoreServices.h>

#include "CloudViewerApp.h"

#include "FileSystem.h"
#include "visualization/gui/Application.h"
#include "visualization/gui/Button.h"
#include "visualization/gui/Dialog.h"
#include "visualization/gui/Label.h"
#include "visualization/gui/Layout.h"
#include "visualization/gui/Native.h"
#include "visualization/gui/Theme.h"
#include "visualization/visualizer/GuiVisualizer.h"

// ----------------------------------------------------------------------------
using namespace cloudViewer::visualization::gui;

class CloudViewerVisualizer : public cloudViewer::visualization::GuiVisualizer {
    using Super = GuiVisualizer;

public:
    CloudViewerVisualizer()
        : cloudViewer::visualization::GuiVisualizer(
                  "CloudViewer", WIDTH, HEIGHT) {
        AddItemsToAppMenu({{"Make Default 3D Viewer", MAC_MAKE_DEFAULT_APP}});
    }

protected:
    static constexpr Menu::ItemId MAC_MAKE_DEFAULT_APP = 100;

    void OnMenuItemSelected(Menu::ItemId item_id) override {
        if (item_id == MAC_MAKE_DEFAULT_APP) {
            auto em = GetTheme().font_size;
            auto dlg = cloudViewer::make_shared<Dialog>(
                    "Make CloudViewer default");

            auto cancel = cloudViewer::make_shared<Button>("Cancel");
            cancel->SetOnClicked([this]() { this->CloseDialog(); });

            auto ok = cloudViewer::make_shared<Button>("Make Default");
            ok->SetOnClicked([this]() {
                // This will set the users personal default to use CloudViewer
                // for the file types below. THIS SHOULD ONLY BE CALLED AFTER
                // THE USER EXPLICITLY CONFIRMS THAT THEY WANT TO DO THIS!
                CFStringRef cloudViewerBundleId =
                        (__bridge CFStringRef) @"com.isl-org.cloudViewer."
                                               @"CloudViewer";
                // The UTIs should match what we declare in Info.plist
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.gl-transmission-format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.gl-binary-transmission-"
                                               @"format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.geometry-definition-"
                                               @"format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.object-file-format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.point-cloud-library-"
                                               @"file",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.polygon-file-format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.3d-points-format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.standard-tesselated-"
                                               @"geometry-format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.xyz-points-format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.xyzn-points-format",
                        kLSRolesAll, cloudViewerBundleId);
                LSSetDefaultRoleHandlerForContentType(
                        (__bridge CFStringRef) @"public.xyzrgb-points-format",
                        kLSRolesAll, cloudViewerBundleId);

                this->CloseDialog();
            });

            auto vert = cloudViewer::make_shared<Vert>(0, Margins(em));
            vert->AddChild(cloudViewer::make_shared<Label>(
                    "This will make CloudViewer the default application for "
                    "the "
                    "following file types:"));
            vert->AddFixed(em);
            auto table =
                    cloudViewer::make_shared<VGrid>(2, 0, Margins(em, 0, 0, 0));
            table->AddChild(cloudViewer::make_shared<Label>("Mesh:"));
            table->AddChild(cloudViewer::make_shared<Label>(
                    ".gltf, .glb, .obj, .off, .ply, .stl"));
            table->AddChild(cloudViewer::make_shared<Label>("Point clouds:"));
            table->AddChild(cloudViewer::make_shared<Label>(
                    ".pcd, .ply, .pts, .xyz, .xyzn, .xyzrgb"));
            vert->AddChild(table);
            vert->AddFixed(em);
            auto buttons = cloudViewer::make_shared<Horiz>(0.5 * em);
            buttons->AddStretch();
            buttons->AddChild(cancel);
            buttons->AddChild(ok);
            vert->AddChild(buttons);
            dlg->AddChild(vert);
            ShowDialog(dlg);
        } else {
            Super::OnMenuItemSelected(item_id);
        }
    }
};

constexpr Menu::ItemId
        CloudViewerVisualizer::MAC_MAKE_DEFAULT_APP;  // for Xcode

// ----------------------------------------------------------------------------
static void LoadAndCreateWindow(const char *path) {
    auto vis = cloudViewer::make_shared<CloudViewerVisualizer>();
    bool is_path_valid = (path && path[0] != '\0');
    if (is_path_valid) {
        vis->LoadGeometry(path);
    }
    Application::GetInstance().AddWindow(vis);
}

// ----------------------------------------------------------------------------
@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@interface AppDelegate () {
    bool open_empty_window_;
}
@property(retain) NSTimer *timer;
@end

@implementation AppDelegate
- (id)init {
    if ([super init]) {
        open_empty_window_ = true;
    }
    return self;
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    // -application:openFile: runs befure applicationDidFinishLaunching: so we
    // need to check if we loaded a file or we need to display an empty window.
    if (open_empty_window_) {
        LoadAndCreateWindow("");
    }
}

// Called by [NSApp run] if the user passes command line arguments (which may
// be multiple times if multiple files are given), or the app is launched by
// double-clicking a file in the Finder, or by dropping files onto the app
// either in the Finder or (more likely) onto the Dock icon. It is also called
// after launching if the user double-clicks a file in the Finder or drops
// a file onto the app icon and the application is already launched.
- (BOOL)application:(NSApplication *)sender openFile:(NSString *)filename {
    open_empty_window_ = false;  // LoadAndCreateWindow() always opens a window
    LoadAndCreateWindow(filename.UTF8String);
    return YES;
}

- (void)applicationWillTerminate:(NSNotification *)notification {
    // The app will terminate after this function exits. This will result
    // in the Application object in main() getting destructed, but it still
    // thinks it is running. So tell Application to quit, which will post
    // the required events to the event loop to properly clean up.
    Application::GetInstance().OnTerminate();
}
@end

// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]) {
    // If we double-clicked the app, the CWD gets set to /, so change that
    // to the user's home directory so that file dialogs act reasonably.
    if (cloudViewer::utility::filesystem::GetWorkingDirectory() == "/") {
        std::string homedir = NSHomeDirectory().UTF8String;
        cloudViewer::utility::filesystem::ChangeWorkingDirectory(homedir);
    }

    Application::GetInstance().Initialize(argc, argv);

    // Note: if NSApp is created (which happens in +sharedApplication)
    //       then GLFW will use our NSApp with our delegate instead of its
    //       own delegate that doesn't have the openFile and terminate
    //       selectors.

    // Ideally we could do the following:
    //@autoreleasepool {
    //    AppDelegate *delegate = [[AppDelegate alloc] init];
    //    NSApplication *application = [NSApplication sharedApplication];
    //    [application setDelegate:delegate];
    //    [NSApp run];
    //}
    // But somewhere along the line GLFW seems to clean up the autorelease
    // pool, which then causes a crash when [NSApp run] finishes and the
    // autorelease pool cleans up at the '}'. To avoid that, we will not
    // autorelease things. That creates a memory leak, but we're exiting
    // after that so it does not matter.
    AppDelegate *delegate = [[AppDelegate alloc] init];
    NSApplication *application = [NSApplication sharedApplication];
    [application setDelegate:delegate];
    // ---- [NSApp run] equivalent ----
    // https://www.cocoawithlove.com/2009/01/demystifying-nsapplication-by.html
    [NSApp finishLaunching];
    Application::GetInstance().Run();
    // ----
}

#endif  // __APPLE__