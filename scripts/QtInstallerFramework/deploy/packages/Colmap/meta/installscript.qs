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
				var exec = "Exec=" + "@TargetDir@/Colmap.sh gui %f" + "\n"; /* Executable command */
				var icon = "Icon=" + "@TargetDir@/Colmap.png" + "\n"; /* Icon resource path */
				var version =  "Version=" + "3.9.0" + "\n" ; /* Version number */
				var name = "Name=" + "Colmap" + "\n"; /* Desktop icon display name */
				var desktop = "Colmap" + ".desktop";  /* Desktop icon file name */
				var comments = "Comment=" + "Structure-from-Motion and Multi-View Stereo" + "\n"
				var comment = name + exec + icon + version + comments + "Terminal=false\nCategories=Graphics\nEncoding=UTF-8\nType=Application\n";
				// crate desktop at @HomeDir@/.local/share/applications
				component.addOperation("CreateDesktopEntry", desktop, comment);
				// crate desktop at @HomeDir@/Desktop
				var desktop_path = "@HomeDir@/Desktop/" + desktop;
				component.addOperation("CreateDesktopEntry", desktop_path, comment);
				component.addOperation("Execute", "sleep", "2");
				component.addOperation("Execute", "bash", "-c", "/usr/bin/gio set \"$0\" metadata::trusted true || true", desktop_path);

			} else if (installer.value("os") === "win") {
				// no need to make application extension files because we can make this on app gui
				// call the base create operations function
				component.addOperation("CreateShortcut",
								"@TargetDir@/Colmap.bat",
								"@StartMenuDir@/Colmap.lnk",
								"workingDirectory=@TargetDir@",
								"description=Open Colmap Application",
                "iconPath=@TargetDir@/Colmap.ico");
				component.addOperation("CreateShortcut",
								"@TargetDir@/Colmap.bat",
								"@DesktopDir@/Colmap.lnk",
								"workingDirectory=@TargetDir@",
								"description=Open Colmap Application",
                "iconPath=@TargetDir@/Colmap.ico");
      }
    } catch (e) {
        console.log(e);
    }
}
