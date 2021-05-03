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
	installer.setInstaller();
	
	// for register file extention
	component.loaded.connect(this, addRegisterFileCheckBox);
    component.unusualFileType = generateUnusualFileType(3);
	
    component.loaded.connect(this, this.installerLoaded);
	
	// for changelog
	installer.installationFinished.connect(this, Component.prototype.installationFinishedPageIsShown);
    installer.finishButtonClicked.connect(this, Component.prototype.installationFinished);
}

Component.prototype.installerLoaded = function()
{
    // installer.setDefaultPageVisible(QInstaller.TargetDirectory, false);
    // installer.addWizardPage(component, "TargetWidget", QInstaller.TargetDirectory);

    // targetDirectoryPage = gui.pageWidgetByObjectName("DynamicTargetWidget");
    QInstaller.windowTitle = "Choose Installation Directory";
    QInstaller.description.setText("Please select where the CloudViewer will be installed:");
    QInstaller.targetDirectory.textChanged.connect(this, this.targetDirectoryChanged);
    QInstaller.targetDirectory.setText(installer.value("TargetDir"));
    QInstaller.targetChooser.released.connect(this, this.targetChooserClicked);
    gui.pageById(QInstaller.ComponentSelection).entered.connect(this, this.componentSelectionPageEntered);
}

Component.prototype.targetChooserClicked = function()
{
    var dir = QFileDialog.getExistingDirectory("", QInstaller.targetDirectory.text);
	if(dir != "")
	{
		QInstaller.targetDirectory.setText(dir);
	}
	else
	{
		QInstaller.targetDirectory.setText(installer.value("TargetDir"));
	}
}

Component.prototype.targetDirectoryChanged = function()
{
    var dir = QInstaller.targetDirectory.text;
    if (installer.fileExists(dir) && installer.fileExists(dir + "/maintenancetool.exe")) {
        QInstaller.warning.setText("<p style=\"color: red\">Existing installation detected and will be overwritten.</p>");
		install.setUpdater();
    }
    else if (installer.fileExists(dir)) {
		QInstaller.warning.setText("<p style=\"color: green\">Installing in existing directory. It will be wiped on uninstallation.</p>");
		installer.setInstaller();
    }
    else {
        QInstaller.warning.setText("");
		installer.setInstaller();
    }
    installer.setValue("TargetDir", dir);
}

Component.prototype.componentSelectionPageEntered = function()
{
    var dir = installer.value("TargetDir");
    if (installer.fileExists(dir) && installer.fileExists(dir + "/maintenancetool.exe")) {
        installer.execute(dir + "/maintenancetool.exe", "--script=" + dir + "/scripts/auto_uninstall.qs");
		install.setUpdater();
    } 
	else
	{
		install.setInstaller();
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
		
		if (installer.value("os") === "win") 
		{
			if (component.userInterface("RegisterFileCheckBoxForm")) {
				var isRegisterFileChecked = component.userInterface("RegisterFileCheckBoxForm").RegisterFileCheckBox.checked;
				if(isRegisterFileChecked)
				{
					var iconId = 0;
					var appPath = "@TargetDir@/ErowCloudViewer.exe";
					for (var i = 0; i < component.unusualFileType.length; i++)
					{
						component.addOperation("RegisterFileType",
										   component.unusualFileType[i],
										   appPath + " %1",
										   "Custom CloudViewer file extension",
										   "text/plain",
										   appPath + "," + iconId,
										   "ProgId=ErowCloudViewer." + component.unusualFileType[i]);
					}
				}
			}

			// call the base create operations function
			component.addOperation("CreateShortcut", 				
				"@TargetDir@/ErowCloudViewer.exe", 
				"@StartMenuDir@/ErowCloudViewer.lnk",
				"workingDirectory=@TargetDir@",
				"description=Open CloudViewer Application");
			component.addOperation("CreateShortcut",
			   "@TargetDir@/ErowCloudViewer.exe",
			   "@DesktopDir@/ErowCloudViewer.lnk",
			   "workingDirectory=@TargetDir@",
			   "description=Open CloudViewer Application");
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
    var extentions = ["bin", "xyz", "ptx", "asc", "xyzrgb", "xyzn", "neu", "txt", "shp", "laz", "las", "ply", "3ds", "fbx", "glb", "ms3d", "dae", "vtk", "csv", "obj", "stl", "off", "pcd", "dxf", "sbf", "e57", "psz", "rdbx", "ptch", "mtch", "vxls", "ifc", "stp", "step", "gltf", "out", "sx", "pn", "pv", "pov", "poly", "mac", "pdms", "pdmsmac"];
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