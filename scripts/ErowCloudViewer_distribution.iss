; �ű��� Inno Setup �ű��� ���ɣ�
; �йش��� Inno Setup �ű��ļ�����ϸ��������İ����ĵ���

#define MyAppName "ErowCloudViewer"
#define MyAppVersion "3.5.0"
#define MyAppPublisher "������Ϣ�Ƽ����޹�˾"
#define MyAppURL "http://www.erow.cn/"
#define MyAppExeName "ErowCloudViewer.exe"

[Setup]
; ע: AppId��ֵΪ������ʶ��Ӧ�ó���
; ��ҪΪ������װ����ʹ����ͬ��AppIdֵ��
; (��Ҫ�����µ� GUID�����ڲ˵��е�� "����|���� GUID"��)
AppId={{0D19DCFB-0F0F-4407-B59B-F53A5016C2FD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; [Icons] �ġ�quicklaunchicon����Ŀʹ�� {userappdata}������ [Tasks] ��Ŀ�����ʺ� IsAdminInstallMode �ļ�顣
UsedUserAreasWarning=no
LicenseFile=G:\develop\pcl_projects\ErowCloudViewer\install\ErowCloudViewer\license.txt
InfoAfterFile=G:\develop\pcl_projects\ErowCloudViewer\install\ErowCloudViewer\global_shift_list_template.txt
; ������ȡ��ע�ͣ����ڷǹ�����װģʽ�����У���Ϊ��ǰ�û���װ����
;PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=C:\Users\Administrator\Desktop
OutputBaseFilename=ErowCloudViewer-3.5.0-2020-08-06-win-amd64
SetupIconFile=G:\develop\pcl_projects\ErowCloudViewer\ErowCloudViewer\eCV\Resources\images\icon\ecv_128.ico
Password=erow2020
Encryption=yes
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
; ע��: ��Ҫ���κι���ϵͳ�ļ���ʹ�á�Flags: ignoreversion��

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
