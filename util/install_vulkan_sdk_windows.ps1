[CmdletBinding()]
param(
    [string]$Version = $(if ($env:VULKAN_SDK_VERSION) { $env:VULKAN_SDK_VERSION } else { "1.4.350.0" }),
    [string]$InstallRoot,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not $InstallRoot) {
    $InstallRoot = Join-Path ${env:ProgramData} "ACloudViewer\VulkanSDK\$Version"
}

$defaultSha256 = "855b27ba05d2d8119c5114c5d4ff870ca38f2c632b11e1bb9923b9b7e6ecfe7b"
$sha256 = if ($Version -eq "1.4.350.0") {
    $defaultSha256
} elseif ($env:VULKAN_SDK_SHA256) {
    $env:VULKAN_SDK_SHA256
} else {
    throw "Set VULKAN_SDK_SHA256 when overriding VULKAN_SDK_VERSION."
}

$glslc = Join-Path $InstallRoot "Bin\glslc.exe"
$spirvHeader = Join-Path $InstallRoot "Include\spirv\unified1\spirv.hpp"
$vulkanLibrary = Join-Path $InstallRoot "Lib\vulkan-1.lib"

if ($Force -or -not (Test-Path $glslc) -or -not (Test-Path $spirvHeader) -or -not (Test-Path $vulkanLibrary)) {
    $workDir = Join-Path ([System.IO.Path]::GetTempPath()) "acloudviewer-vulkan-$Version"
    $installer = Join-Path $workDir "vulkan_sdk.exe"
    Remove-Item $workDir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $workDir | Out-Null

    $uri = "https://sdk.lunarg.com/sdk/download/$Version/windows/vulkan_sdk.exe"
    Write-Host "Downloading Vulkan SDK $Version from $uri"
    Invoke-WebRequest -Uri $uri -OutFile $installer
    $actualSha256 = (Get-FileHash -Algorithm SHA256 $installer).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $sha256.ToLowerInvariant()) {
        throw "Vulkan SDK SHA256 mismatch: got $actualSha256, expected $sha256"
    }

    $arguments = @(
        "--root", $InstallRoot,
        "--accept-licenses",
        "--default-answer",
        "--confirm-command",
        "install",
        "copy_only=1"
    )
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru -NoNewWindow
    if ($process.ExitCode -ne 0) {
        throw "Vulkan SDK installer failed with exit code $($process.ExitCode)"
    }
    Remove-Item $workDir -Recurse -Force
}

foreach ($path in @($glslc, $spirvHeader, $vulkanLibrary)) {
    if (-not (Test-Path $path)) {
        throw "Incomplete Vulkan SDK installation; missing $path"
    }
}

$env:VULKAN_SDK = $InstallRoot
$env:PATH = "$(Join-Path $InstallRoot 'Bin');$env:PATH"

if ($env:GITHUB_ENV) {
    "VULKAN_SDK=$InstallRoot" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
if ($env:GITHUB_PATH) {
    (Join-Path $InstallRoot "Bin") | Out-File -FilePath $env:GITHUB_PATH -Encoding utf8 -Append
}

Write-Host "Vulkan SDK ready: $InstallRoot"
Write-Host "  glslc: $glslc"
Write-Host "  SPIR-V headers: $spirvHeader"
Write-Host "  import library: $vulkanLibrary"
& $glslc --version
