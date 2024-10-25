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
    targetDirectoryPage.description.setText("Please select where the CloudViewer will be installed:");
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
    var dir = targetDirectoryPage.targetDirectory.text;
    if (installer.fileExists(dir) && installer.fileExists(dir + "/maintenancetool")) {
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
    var dir = installer.value("TargetDir");
    if (installer.fileExists(dir) && installer.fileExists(dir + "/maintenancetool")) {
        installer.execute(dir + "/maintenancetool", "--script=" + dir + "/scripts/auto_uninstall.qs");
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
		
		if (installer.value("os") === "x11") 
		{
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
			var exec = "Exec=" + "@TargetDir@/CloudViewer.sh %f" + "\n"; /* 执行程序 */
			var icon = "Icon=" + "@TargetDir@/CloudViewer.png" + "\n"; /* 图标资源路径 */
			var version =  "Version=" + "3.9.0" + "\n" ; /* 版本号 */
			var name = "Name=" + "CloudViewer" + "\n"; /* 桌面图标显示名称 */
			var desktop = "CloudViewer" + ".desktop";  /* 桌面图标名 */
			var comments = "Comment=" + "3D point cloud and mesh processing software" + "\n"
			var extentions = "MimeType=" + "model/stl;model/obj;model/fbx;model/gltf-binary;model/gltf+json;model/x.stl-ascii;model/x.stl-binary;model/x-ply;application/x-off;application/x-xyz;application/x-xyzn;application/x-xyzrgb;application/x-pcd;application/x-pts" + "\n"
			var comment = name + exec + icon + version + comments + "Terminal=false\nCategories=Graphics\nEncoding=UTF-8\nType=Application\n" + extentions;
			component.addOperation("CreateDesktopEntry", desktop, comment);

			//获取当前桌面路径    
			var desktoppath = QDesktopServices.storageLocation(0);
			var homeDir = installer.environmentVariable("HOME");
		    if(homeDir.length > 0) {
                var deskTopSaveDir = homeDir + "/.local/share/applications/" + desktop;
                if(installer.fileExists(deskTopSaveDir)) {
                    var args = ["cp", "-R", deskTopSaveDir, desktoppath + "/" + desktop];
                    component.addOperation("Execute", args);
                }
            }
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
