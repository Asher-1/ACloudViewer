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
				/***************************************路径说明****************************************
				系统自带变量
				TargetDir   目标安装目录，由用户选择
				DesktopDir  用户桌面目录名(路径)。仅在Windows上可用
				RootDir 文件系统根目录
				HomeDir 当前用户的home目录
				ApplicationsDir 应用程序目录。例如,Windows上的C:\Program Files,Linux上/opt以及OS X上/Applications
				InstallerDirPath    包含安装程序可执行文件的目录
				InstallerFilePath   安装程序可执行文件的文件路径
				
				注意：变量是包含在“@@”中的，以@开始，必须要以@结尾
				
				具体的其它信息可以参考 https://www.cnblogs.com/oloroso/p/6775318.html#7_3_2_3
				**************************************************************************************/        
				/* 建立桌面图标 */
				var exec = "Exec=" + "@TargetDir@/Colmap.sh gui %f" + "\n"; /* 执行程序 */
				var icon = "Icon=" + "@TargetDir@/Colmap.png" + "\n"; /* 图标资源路径 */
				var version =  "Version=" + "3.9.0" + "\n" ; /* 版本号 */
				var name = "Name=" + "Colmap" + "\n"; /* 桌面图标显示名称 */
				var desktop = "Colmap" + ".desktop";  /* 桌面图标名 */
				var comments = "Comment=" + "Structure-from-Motion and Multi-View Stereo" + "\n"
				var comment = name + exec + icon + version + comments + "Terminal=false\nCategories=Graphics\nEncoding=UTF-8\nType=Application\n";
				// crate desktop at @HomeDir@/.local/share/applications
				component.addOperation("CreateDesktopEntry", desktop, comment);
				// crate desktop at @HomeDir@/Desktop
				var desktop_path = "@HomeDir@/Desktop/" + desktop;
				component.addOperation("CreateDesktopEntry", desktop_path, comment);
				component.addOperation("Execute", "sleep", "2");
				component.addOperation("Execute", "/usr/bin/gio", "set", desktop_path, "metadata::trusted", "true");

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
