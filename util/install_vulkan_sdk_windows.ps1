[CmdletBinding()]
param(
    [string]$Version = $(if ($env:VULKAN_SDK_VERSION) { $env:VULKAN_SDK_VERSION } else { "1.4.350.0" }),
    [string]$InstallRoot,
    [switch]$Force,
    [switch]$SkipProfile
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$syncScript = Join-Path $scriptDir "sync_vulkan_env_from_sdk.ps1"

function Test-VulkanSdkTree {
    param([string]$Root)

    if (-not $Root) { return $false }
    foreach ($sdkRoot in @($Root, (Join-Path $Root "x86_64"))) {
        foreach ($includeDir in @("Include", "include")) {
            $header = Join-Path $sdkRoot "$includeDir\vulkan\vulkan_core.h"
            if (-not (Test-Path $header)) { continue }
            foreach ($binDir in @("Bin", "bin")) {
                if (Test-Path (Join-Path $sdkRoot "$binDir\glslc.exe")) {
                    return $true
                }
            }
        }
    }
    return $false
}

if ($env:VULKAN_SDK -and (Test-VulkanSdkTree $env:VULKAN_SDK)) {
    Write-Host "Using preinstalled Vulkan SDK at $($env:VULKAN_SDK)"
    if ($SkipProfile) {
        & $syncScript -SkipProfile
    } else {
        & $syncScript
    }
    return
}

if (-not $InstallRoot) {
    $InstallRoot = Join-Path ${env:ProgramData} "ACloudViewer\VulkanSDK\$Version"
}

if ((Test-VulkanSdkTree $InstallRoot) -and -not $Force) {
    $env:VULKAN_SDK = $InstallRoot
    Write-Host "Reusing existing Vulkan SDK at $InstallRoot"
    if ($SkipProfile) {
        & $syncScript -SkipProfile
    } else {
        & $syncScript
    }
    return
}

$defaultSha256 = "855b27ba05d2d8119c5114c5d4ff870ca38f2c632b11e1bb9923b9b7e6ecfe7b"
$sha256 = if ($Version -eq "1.4.350.0") {
    $defaultSha256
} elseif ($env:VULKAN_SDK_SHA256) {
    $env:VULKAN_SDK_SHA256
} else {
    throw "Set VULKAN_SDK_SHA256 when overriding VULKAN_SDK_VERSION."
}

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

$env:VULKAN_SDK = $InstallRoot
if ($SkipProfile -or $env:GITHUB_ACTIONS) {
    & $syncScript -SkipProfile
} else {
    & $syncScript
}
