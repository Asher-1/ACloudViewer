/****************************************************************************
**
** Copyright (C) 2017 The Qt Company Ltd.
** Contact: https://www.qt.io/licensing/
**
** This file is part of the FOO module of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:GPL-EXCEPT$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3 as published by the Free Software
** Foundation with exceptions as appearing in the file LICENSE.GPL3-EXCEPT
** included in the packaging of this file. Please review the following
** information to ensure the GNU General Public License requirements will
** be met: https://www.gnu.org/licenses/gpl-3.0.html.
**
** $QT_END_LICENSE$
**
****************************************************************************/

var targetDirectoryPage = null;

function Component()
{
    // for autoinstall
    component.loaded.connect(this, this.installerLoaded);

    // for register file extention
    component.loaded.connect(this, addRegisterFileCheckBox);
    component.unusualFileType = generateUnusualFileType(3);
	
    // for changelog
    installer.installationFinished.connect(this, Component.prototype.installationFinishedPageIsShown);
    installer.finishButtonClicked.connect(this, Component.prototype.installationFinished);
}

Component.prototype.installerLoaded = function()
{
    installer.setDefaultPageVisible(QInstaller.TargetDirectory, false);
    installer.addWizardPage(component, "TargetWidget", QInstaller.TargetDirectory);

    targetDirectoryPage = gui.pageWidgetByObjectName("DynamicTargetWidget");
    targetDirectoryPage.windowTitle = "Choose Installation Directory";
    targetDirectoryPage.description.setText("Please select where the ACloudViewer will be installed:");
    targetDirectoryPage.targetDirectory.textChanged.connect(this, this.targetDirectoryChanged);
    targetDirectoryPage.targetDirectory.setText(installer.value("TargetDir"));
    targetDirectoryPage.targetChooser.released.connect(this, this.targetChooserClicked);
    gui.pageById(QInstaller.ComponentSelection).entered.connect(this, this.componentSelectionPageEntered);
}

Component.prototype.targetChooserClicked = function()
{
	var dir = QFileDialog.getExistingDirectory("", targetDirectoryPage.targetDirectory.text);
	if(dir != "") {
		targetDirectoryPage.targetDirectory.setText(dir);
	} else {
		targetDirectoryPage.targetDirectory.setText(installer.value("TargetDir"));
	}
}

Component.prototype.targetDirectoryChanged = function()
{
    var maintenance_name;
    if (installer.value("os") == "mac") {
       maintenance_name  = "/maintenancetool.app";    
	} else if (installer.value("os") === "x11") {
       maintenance_name  = "/maintenancetool";
    } else if (installer.value("os") === "win") {
        maintenance_name  = "/maintenancetool.exe";
    }
    var dir = targetDirectoryPage.targetDirectory.text;
    if (installer.fileExists(dir) && installer.fileExists(dir + maintenance_name)) {
        targetDirectoryPage.warning.setText("<p style=\"color: red\">Existing installation detected and will be overwritten.</p>");
    } else if (installer.fileExists(dir)) {
		targetDirectoryPage.warning.setText("<p style=\"color: green\">Installing in existing directory. It will be wiped on uninstallation.</p>");
    } else {
        targetDirectoryPage.warning.setText("");
    }
    installer.setValue("TargetDir", dir);
}

Component.prototype.componentSelectionPageEntered = function()
{
    var maintenance_name;
    if (installer.value("os") == "mac") {
       maintenance_name  = "/maintenancetool.app";
	} else if (installer.value("os") === "x11") {
       maintenance_name  = "/maintenancetool";
    } else if (installer.value("os") === "win") {
       maintenance_name  = "/maintenancetool.exe";
    }
    var dir = installer.value("TargetDir");
    if (installer.fileExists(dir) && installer.fileExists(dir + maintenance_name)) {
        installer.execute(dir + maintenance_name, "--script=" + dir + "/scripts/auto_uninstall.qs");
    }
}

Component.prototype.isDefault = function()
{
    // select the component by default
    return true;
}

Component.prototype.createOperations = function()
{
    try {
		// call default implementation to actually install the registeredfile
		component.createOperations();

        if (installer.value("os") == "mac") {
            // no need to make shortcut on macos
		} else if (installer.value("os") === "x11") {
			/*************************************** Path variable reference ****************************************
			Built-in variables
			TargetDir   Target installation directory, chosen by the user
			DesktopDir  User desktop directory name (path). Available on Windows only
			RootDir     File system root directory
			HomeDir     Current user's home directory
			ApplicationsDir  Applications directory. e.g. C:\Program Files on Windows, /opt on Linux, /Applications on OS X
			InstallerDirPath    Directory containing the installer executable
			InstallerFilePath   File path of the installer executable
			
			Note: Variables are enclosed in "@@", starting with @ and must end with @
			
			For more details see https://www.cnblogs.com/oloroso/p/6775318.html#7_3_2_3
			**************************************************************************************/        
			/* Create desktop icon */
			var exec = "Exec=" + "@TargetDir@/ACloudViewer.sh %f" + "\n"; /* Executable command */
			var icon = "Icon=" + "@TargetDir@/ACloudViewer.png" + "\n"; /* Icon resource path */
			var version =  "Version=" + "3.9.0" + "\n" ; /* Version number */
			var name = "Name=" + "ACloudViewer" + "\n"; /* Desktop icon display name */
			var desktop = "ACloudViewer" + ".desktop";  /* Desktop icon file name */
			var comments = "Comment=" + "3D point cloud and mesh processing software" + "\n"
			var comment = name + exec + icon + version + comments + "Terminal=false\nCategories=Graphics\nEncoding=UTF-8\nType=Application\n";
            // crate desktop at @HomeDir@/.local/share/applications
            component.addOperation("CreateDesktopEntry", desktop, comment);
            // crate desktop at @HomeDir@/Desktop
            var desktop_path = "@HomeDir@/Desktop/" + desktop;
            component.addOperation("CreateDesktopEntry", desktop_path, comment);
            component.addOperation("Execute", "sleep", "2");
            component.addOperation("Execute", "bash", "-c", "/usr/bin/gio set \"$0\" metadata::trusted true || true", desktop_path);

		} else if (installer.value("os") === "win") {
            // register extensions support
            if (component.userInterface("RegisterFileCheckBoxForm")) {
                var isRegisterFileChecked = component.userInterface("RegisterFileCheckBoxForm").RegisterFileCheckBox.checked;
                if(isRegisterFileChecked) {
                    var iconId = 0;
                    var appPath  = "@TargetDir@/ACloudViewer.exe";
                    for (var i = 0; i < component.unusualFileType.length; i++) {
                        component.addOperation("RegisterFileType",
                                                            component.unusualFileType[i],
                                                            appPath + " %1",
                                                            "Custom ACloudViewer file extension",
                                                            "text/plain",
                                                            appPath + "," + iconId,
                                                            "ProgId=ACloudViewer." + component.unusualFileType[i]);
                    }
                }
            }
            
            // call the base create operations function
            component.addOperation("CreateShortcut",
                    "@TargetDir@/ACloudViewer.bat",
                    "@StartMenuDir@/ACloudViewer.lnk",
                    "workingDirectory=@TargetDir@",
                    "description=Open ACloudViewer Application",
                    "iconPath=@TargetDir@/ACloudViewer.ico");
            component.addOperation("CreateShortcut",
                    "@TargetDir@/ACloudViewer.bat",
                    "@DesktopDir@/ACloudViewer.lnk",
                    "workingDirectory=@TargetDir@",
                    "description=Open ACloudViewer Application",
                    "iconPath=@TargetDir@/ACloudViewer.ico");
        }
    } catch (e) {
        console.log(e);
    }
}

Component.prototype.installationFinishedPageIsShown = function()
{
    try {
        if (installer.isInstaller() && installer.status == QInstaller.Success) {
            installer.addWizardPageItem( component, "ReadMeCheckBoxForm", QInstaller.InstallationFinished );
        }
    } catch(e) {
        console.log(e);
    }
}

Component.prototype.installationFinished = function()
{
    try {
        if (installer.isInstaller() && installer.status == QInstaller.Success) {
            var checkboxForm = component.userInterface( "ReadMeCheckBoxForm" );
            if (checkboxForm && checkboxForm.readMeCheckBox.checked) {
                QDesktopServices.openUrl("file:///" + installer.value("TargetDir") + "/CHANGELOG.txt");
            }
        }
    } catch(e) {
        console.log(e);
    }
}

generateUnusualFileType = function(length)
{
    var extentions = ["bin", "xyz", "ptx", "asc", "xyzrgb", "xyzn", "neu", "txt", "shp", "laz", "las", "ply", "3ds",
                      "fbx", "glb", "ms3d", "dae", "vtk", "csv", "obj", "stl", "off", "pcd", "dxf", "sbf", "e57", "psz",
                      "rdbx", "ptch", "mtch", "vxls", "ifc", "stp", "step", "gltf", "out", "sx", "pn", "pv", "pov", "poly",
                      "mac", "pdms", "pdmsmac"];
    return extentions;
}

// called as soon as the component was loaded
addRegisterFileCheckBox = function()
{
    if (installer.isInstaller())
    {
        if (installer.addWizardPageItem(component, "RegisterFileCheckBoxForm", QInstaller.TargetDirectory))
        {
            var extensions = "\n";
            for (var i = 0; i < component.unusualFileType.length; i++)
            {
                if(i % 10 == 0)
                {
                        extensions += "\n";
                }
                else
                {
                        extensions += ", ";
                }
                extensions += component.unusualFileType[i];
            }
            component.userInterface("RegisterFileCheckBoxForm").RegisterFileCheckBox.text =
                    component.userInterface("RegisterFileCheckBoxForm").RegisterFileCheckBox.text + extensions;
        }
        else
        {
                console.log("Could not add the dynamic page item.");
        }
    }
}
