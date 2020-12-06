; 脚本由 Inno Setup 脚本向导 生成！
; 有关创建 Inno Setup 脚本文件的详细资料请查阅帮助文档！

#define MyAppName "ErowCloudViewer"
#define MyAppVersion "3.7.0"
#define MyAppPublisher "逸舟信息科技有限公司"
#define MyAppURL "http://www.erow.cn/"
#define MyAppExeName "ErowCloudViewer.exe"

[Setup]
; 注: AppId的值为单独标识该应用程序。
; 不要为其他安装程序使用相同的AppId值。
; (若要生成新的 GUID，可在菜单中点击 "工具|生成 GUID"。)
AppId={{7B525361-65B2-4585-B196-F6CBF0145A5C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; [Icons] 的“quicklaunchicon”条目使用 {userappdata}，而其 [Tasks] 条目具有适合 IsAdminInstallMode 的检查。
UsedUserAreasWarning=no
LicenseFile=G:\develop\pcl_projects\ErowCloudViewer\install\ErowCloudViewer\license.txt
InfoAfterFile=G:\develop\pcl_projects\ErowCloudViewer\install\ErowCloudViewer\global_shift_list_template.txt
; 以下行取消注释，以在非管理安装模式下运行（仅为当前用户安装）。
;PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=C:\Users\Administrator\Desktop
OutputBaseFilename=ErowCloudViewer-3.6.0-2020-10-12-win-amd64
SetupIconFile=G:\develop\pcl_projects\ErowCloudViewer\ErowCloudViewer\eCV\Resources\images\icon\ecv_128.ico
Password=
Encryption=no
Compression=lzma
SolidCompression=yes
WizardStyle=modern

 [Dirs]
Name: {app}; Permissions: users-full

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
Source: "G:\develop\pcl_projects\ErowCloudViewer\install\ErowCloudViewer\ErowCloudViewer.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "G:\develop\pcl_projects\ErowCloudViewer\install\ErowCloudViewer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 注意: 不要在任何共享系统文件上使用“Flags: ignoreversion”

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

